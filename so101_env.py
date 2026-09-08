#!/usr/bin/env python3
"""
so101_env.py — Gymnasium GoalEnv for SO-101 Ball Picker in MuJoCo.

Reward: "V11" staged reward, adapted from ggando's published SO-101 MuJoCo/SAC
grasp-and-lift agent (https://ggando.com/blog/so101-rl-lift/, code at
https://github.com/ggand0/pick-101), which reported 100% success rate with
this structure. Per-step sum of terms (NOT a potential-based delta like the
old OpenAI-Fetch-style shaping this replaces):

  Component            Condition          Value
  --------------------------------------------------------------------------
  Reach                always             1.0 - tanh(10 * dist_to_ball)
  Push-down penalty    ball_z < 0.010     -(0.010 - ball_z) * 50
  Drop penalty         lost grasp*        -2.0
  Grasp bonus          grasping**         +0.25
  Continuous lift      grasping**         lift_progress * 2.0
  Binary lift          ball_z > 0.040     +1.0
  Target bonus         ball_z > TARGET_Z  +1.0
  Action penalty       ball_z > 0.060     -0.01 * ||action - prev_action||^2
  Success              held at target***  +10.0

  *   "lost grasp" = was touching AND lifted last step, not touching this step.
  **  "grasping" = both jaw pads touching the ball simultaneously (is_touching).
  *** "held at target" = grasping AND ball_z > TARGET_Z, sustained for a few
      consecutive steps (see SUCCESS_HOLD_STEPS) so a single noisy frame can't
      end an episode.

Two height thresholds were rescaled from the blog's original numbers (0.02
binary-lift / 0.08 target) to fit THIS env's ball, which rests at ~0.024m
instead of their cube's lower resting height — see BINARY_LIFT_Z / TARGET_Z
below. The push-down threshold (0.010) and action-penalty threshold (0.060)
were kept as published since they're near-table / near-target checks that
don't depend much on the object's exact resting height.

Also keeps the STABLE_GRASP_BONUS anti-flick check added 2026-09-08 (ball
must move WITH the gripper, not on its own separate flight path, for the
hold to count) — that's this project's own addition on top of V11, not part
of the original blog post.

Compatible with:
  • Stable-Baselines3 HerReplayBuffer (HER) — requires GoalEnv dict observation space.
    NOTE: V11 doesn't depend on desired_goal (it's a fixed-height staged reward,
    not a goal-distance reward), so HER's goal-relabeling trick isn't doing
    useful work under V11 the way it did under the old goal-distance shaping.
    The Dict observation / achieved_goal / desired_goal / compute_reward
    interface is kept as-is so the existing training script needs no changes.
  • EvalCallback, CheckpointCallback
  • Curriculum level control via set_curriculum_level(1/2/3)
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "so101_mujoco.xml")

# ── V11 reward constants (see module docstring table) ───────────────────────
PUSH_DOWN_Z      = 0.010   # published as-is: near/through-the-table check
BINARY_LIFT_Z    = 0.040   # rescaled from blog's 0.02 — this ball rests at ~0.024,
                            # so 0.02 would fire even at rest; 0.04 is genuinely lifted
ACTION_PENALTY_Z = 0.060   # published as-is
TARGET_Z         = 0.080   # rescaled from blog's 0.08 target — kept the same number,
                            # it already sits sensibly above this env's resting height
REACH_TANH_SCALE      = 10.0
PUSH_DOWN_COEF        = 50.0
DROP_PENALTY          = -2.0
GRASP_BONUS            = 0.25
CONTINUOUS_LIFT_COEF  = 2.0
BINARY_LIFT_BONUS     = 1.0
TARGET_BONUS          = 1.0
ACTION_PENALTY_COEF   = 0.01
SUCCESS_BONUS         = 10.0
SUCCESS_HOLD_STEPS    = 3   # consecutive steps "held at target" must hold before it counts

# Added 2026-09-08, kept on top of V11: a big one-time bonus the instant it
# genuinely grasps the ball AND lifts it AND holds it there continuously
# without dropping it, moving together with the gripper (not flicked/tossed).
STABLE_HOLD_STEPS    = 15     # ~0.3s of unbroken, genuine hold while lifted
STABLE_LIFT_HEIGHT   = 0.06   # ball must be at least 6cm off the table
STABLE_GRASP_BONUS   = 300.0
STABLE_MAX_REL_SPEED = 0.25   # m/s, ball velocity relative to gripper velocity


class SO101PickEnv(gym.Env):
    """
    Gymnasium GoalEnv for SO-101 robot arm picking an orange foam ball.
    Reward is the V11 staged reward (see module docstring) — reward_type is
    kept as a constructor arg for interface compatibility but only "v11" is
    implemented now; "sparse"/"dense"/"shaped" (the old OpenAI-Fetch-style
    reward) have been removed.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, reward_type="v11"):
        super().__init__()
        self.render_mode = render_mode
        if reward_type != "v11":
            raise ValueError(f"Only reward_type='v11' is supported now (got {reward_type!r}).")
        self.reward_type = reward_type

        if not os.path.exists(XML_PATH):
            raise FileNotFoundError(f"MuJoCo XML model not found at {XML_PATH}")

        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Joint limits
        self.joint_min = self.model.jnt_range[:6, 0]
        self.joint_max = self.model.jnt_range[:6, 1]

        # Action: 6 motor delta angle increments [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # ── GoalEnv Dict Observation (required for HER compatibility) ──────────
        obs_dim = 21  # 6 qpos + 6 qvel + 3 gripper_pos + 3 ball_pos + 3 rel_vec
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-10.0, 10.0, shape=(3,),      dtype=np.float32),
            "desired_goal":  spaces.Box(-10.0, 10.0, shape=(3,),      dtype=np.float32),
        })

        # Body / Geom IDs
        self.gripper_body_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1_link")
        self.ball_body_id     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.ball_jnt_id      = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos_adr    = self.model.jnt_qposadr[self.ball_jnt_id]
        self.ball_dof_adr     = self.model.jnt_dofadr[self.ball_jnt_id]
        self.ball_geom_id     = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.table_geom_id    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_solid")
        self.fixed_jaw_geom_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_collider")
        self.moving_jaw_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_collider")

        self.n_substeps  = 10    # 10 * 0.002s = 0.02s per step (50 Hz control — same as OpenAI Fetch)
        self.action_scale = 0.06
        self.max_steps   = 350
        self.current_step = 0
        self.held_in_air_steps = 0

        # Home + curriculum
        self.home_rad = np.deg2rad(np.array([0.0, -50.0, 50.0, 60.0, 0.0, 25.0], dtype=np.float32))
        self.curriculum_level = 3

        # Goal (set during reset) — kept for GoalEnv/HER interface compatibility.
        # V11's reward doesn't use this (it's a fixed-height staged reward, not
        # a goal-distance reward), but the Dict obs space still needs the key.
        self._goal = np.zeros(3, dtype=np.float32)

    # ── Curriculum helpers ────────────────────────────────────────────────────

    def set_curriculum_level(self, level):
        """1 = Lift Only, 2 = Drop & Clamp, 3 = Full Workspace."""
        self.curriculum_level = int(np.clip(level, 1, 3))

    # ── Utility ───────────────────────────────────────────────────────────────

    def _get_pinch_pos(self):
        return (self.data.geom_xpos[self.fixed_jaw_geom_id] +
                self.data.geom_xpos[self.moving_jaw_geom_id]) / 2.0

    def _is_contacting(self):
        """True only if BOTH inner rubber jaw pads touch the ball simultaneously.
        This is what V11's "grasping" condition maps to in this env."""
        fixed_touch  = False
        moving_touch = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.ball_geom_id in (g1, g2):
                other = g2 if g1 == self.ball_geom_id else g1
                if other == self.fixed_jaw_geom_id:
                    fixed_touch = True
                if other == self.moving_jaw_geom_id:
                    moving_touch = True
        return fixed_touch and moving_touch   # BOTH pads must touch simultaneously

    # ── GoalEnv-compatible compute_reward ────────────────────────────────────
    # Signature required by SB3's HerReplayBuffer. V11's value is precomputed,
    # goal-independent, in step() and just passed through info — it doesn't
    # actually depend on achieved_goal/desired_goal (see module docstring).

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._info_field(info, "v11_reward", 0.0).astype(np.float32)

    @staticmethod
    def _info_field(info, key, default):
        """Pull `key` out of `info`, whether it's a single dict (a live
        env.step() call) or an array/list of dicts (HER's batched calls)."""
        if isinstance(info, dict):
            return np.asarray(info.get(key, default))
        return np.asarray([d.get(key, default) if isinstance(d, dict) else default for d in info])

    # ── Observation builder ───────────────────────────────────────────────────

    def _get_obs(self):
        qpos        = self.data.qpos[:6].astype(np.float32)
        qvel        = self.data.qvel[:6].astype(np.float32)
        gripper_pos = self._get_pinch_pos().astype(np.float32)
        ball_pos    = self.data.xpos[self.ball_body_id].astype(np.float32)
        rel_pos     = (ball_pos - gripper_pos).astype(np.float32)

        obs_vec = np.clip(np.concatenate([qpos, qvel, gripper_pos, ball_pos, rel_pos]),
                          -10.0, 10.0).astype(np.float32)

        return {
            "observation":   obs_vec,
            "achieved_goal": ball_pos.copy(),       # current ball XYZ
            "desired_goal":  self._goal.copy(),     # kept for interface compatibility (unused by V11)
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step      = 0
        self.held_in_air_steps = 0
        self._stable_bonus_given = False   # resets each episode
        self._stable_grip_steps  = 0
        self._prev_pinch_pos     = self._get_pinch_pos().copy()
        self._prev_action        = np.zeros(6, dtype=np.float32)
        self._was_touching_lifted = False
        self._success_hold_steps  = 0

        # Randomize ball position on desk
        bx = self.np_random.uniform(0.18, 0.23)
        by = self.np_random.uniform(-0.07, 0.07)
        bz = 0.024
        base_angle = float(np.arctan2(by, bx))

        self.data.qpos[self.ball_qpos_adr     : self.ball_qpos_adr + 3] = [bx, by, bz]
        self.data.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Goal: kept for GoalEnv/HER interface compatibility (see class docstring)
        self._goal = np.array([bx, by, TARGET_Z], dtype=np.float32)

        # Curriculum arm spawn pose
        if self.curriculum_level == 1:
            pose = np.array([base_angle, np.deg2rad(22.0), np.deg2rad(18.0), np.deg2rad(50.0), 0.0, np.deg2rad(25.0)])
        elif self.curriculum_level == 2:
            pose = np.array([base_angle, np.deg2rad(5.0),  np.deg2rad(35.0), np.deg2rad(50.0), 0.0, np.deg2rad(25.0)])
        else:
            pose = self.home_rad.copy()

        jitter       = self.np_random.uniform(-0.02, 0.02, size=6)
        start_qpos   = np.clip(pose + jitter, self.joint_min, self.joint_max)
        self.data.qpos[:6] = start_qpos
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = start_qpos

        mujoco.mj_forward(self.model, self.data)

        self._ball_start_z    = float(bz)
        self._gripper_start_z = float(self._get_pinch_pos()[2])

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Floor safety: prevent pushing arm into table
        if self._get_pinch_pos()[2] < 0.025:
            action[1] = min(action[1], 0.0)
            action[2] = max(action[2], 0.0)

        new_ctrl = self.data.ctrl[:6] + action * self.action_scale
        new_ctrl[0] = np.clip(new_ctrl[0], np.deg2rad(-65.0), np.deg2rad(65.0))
        new_ctrl[1] = np.clip(new_ctrl[1], np.deg2rad(-75.0), np.deg2rad(30.0))
        new_ctrl[2] = np.clip(new_ctrl[2], np.deg2rad(-10.0), self.joint_max[2])
        self.data.ctrl[:6] = np.clip(new_ctrl, self.joint_min, self.joint_max)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Ball stability: dampen micro-drift when untouched on table
        ball_pos = self.data.xpos[self.ball_body_id]
        if not self._is_contacting() and ball_pos[2] < 0.032:
            if np.linalg.norm(self.data.qvel[self.ball_dof_adr:self.ball_dof_adr+3]) < 0.05:
                self.data.qvel[self.ball_dof_adr:self.ball_dof_adr+6] = 0.0

        is_touching  = self._is_contacting()   # V11's "grasping" signal
        gripper_pos  = self._get_pinch_pos()
        ball_pos     = self.data.xpos[self.ball_body_id].copy().astype(np.float32)
        dist_to_ball = float(np.linalg.norm(ball_pos - gripper_pos))
        ball_z       = float(ball_pos[2])
        grip_angle   = float(self.data.qpos[5])

        # ── V11 staged reward (see module docstring table) ─────────────────────
        reach_term       = 1.0 - float(np.tanh(REACH_TANH_SCALE * dist_to_ball))
        push_down_term    = -(PUSH_DOWN_Z - ball_z) * PUSH_DOWN_COEF if ball_z < PUSH_DOWN_Z else 0.0
        lifted_now         = is_touching and ball_z > BINARY_LIFT_Z
        drop_term          = DROP_PENALTY if (self._was_touching_lifted and not lifted_now) else 0.0
        grasp_term         = GRASP_BONUS if is_touching else 0.0
        lift_progress      = float(np.clip((ball_z - self._ball_start_z) / (TARGET_Z - self._ball_start_z), 0.0, 1.0))
        continuous_lift_term = CONTINUOUS_LIFT_COEF * lift_progress if is_touching else 0.0
        binary_lift_term  = BINARY_LIFT_BONUS if ball_z > BINARY_LIFT_Z else 0.0
        target_term       = TARGET_BONUS if ball_z > TARGET_Z else 0.0
        action_delta       = action - self._prev_action
        action_penalty_term = -ACTION_PENALTY_COEF * float(np.dot(action_delta, action_delta)) if ball_z > ACTION_PENALTY_Z else 0.0

        held_at_target = is_touching and ball_z > TARGET_Z
        self._success_hold_steps = (self._success_hold_steps + 1) if held_at_target else 0
        success_term = SUCCESS_BONUS if self._success_hold_steps >= SUCCESS_HOLD_STEPS else 0.0

        self._was_touching_lifted = lifted_now
        self._prev_action = action.copy()

        # Sustained hold tracking (used elsewhere, e.g. curriculum success rate)
        is_held = bool(is_touching and dist_to_ball <= 0.06 and grip_angle < 0.28)
        if is_held and ball_z > 0.040:
            self.held_in_air_steps += 1
        else:
            self.held_in_air_steps = 0

        # ── Anti-flick stable-grasp bonus (this project's own addition on top
        # of V11, added 2026-09-08) — ball must move WITH the gripper, not on
        # its own separate flight path, for the hold streak to count.
        dt          = self.n_substeps * self.model.opt.timestep
        gripper_vel = (gripper_pos - self._prev_pinch_pos) / dt
        ball_vel    = self.data.qvel[self.ball_dof_adr:self.ball_dof_adr + 3]
        rel_speed   = float(np.linalg.norm(ball_vel - gripper_vel))
        self._prev_pinch_pos = gripper_pos.copy()

        genuinely_carried = (is_held and ball_z >= STABLE_LIFT_HEIGHT
                              and rel_speed <= STABLE_MAX_REL_SPEED)
        if genuinely_carried:
            self._stable_grip_steps += 1
        else:
            self._stable_grip_steps = 0

        stable_grasp_bonus = 0.0
        if not self._stable_bonus_given and self._stable_grip_steps >= STABLE_HOLD_STEPS:
            stable_grasp_bonus = STABLE_GRASP_BONUS
            self._stable_bonus_given = True

        v11_reward = (reach_term + push_down_term + drop_term + grasp_term
                      + continuous_lift_term + binary_lift_term + target_term
                      + action_penalty_term + success_term + stable_grasp_bonus)

        achieved_goal = ball_pos
        desired_goal  = self._goal
        info = {
            "dist":              dist_to_ball,
            "ball_z":            ball_z,
            "is_touching":       is_touching,
            "v11_reward":        v11_reward,
            "is_held":           is_held,
            "held_in_air_steps": self.held_in_air_steps,
            "grip_angle":        grip_angle,
        }
        reward = float(self.compute_reward(achieved_goal, desired_goal, info))

        # Episode success: genuinely held at/above target height for a few
        # consecutive steps (same condition that unlocks the V11 success_term).
        success    = bool(self._success_hold_steps >= SUCCESS_HOLD_STEPS)
        terminated = success
        truncated  = bool(self.current_step >= self.max_steps)

        obs = self._get_obs()
        info["success"]    = success
        info["is_success"] = success   # SB3 HER reads "is_success" key

        return obs, reward, terminated, truncated, info

    def render(self):
        pass
