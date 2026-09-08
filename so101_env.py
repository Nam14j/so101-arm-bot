#!/usr/bin/env python3
"""
so101_env.py — Gymnasium GoalEnv for SO-101 Ball Picker in MuJoCo.

Reward equation from OpenAI FetchPickAndPlace (Plappert et al., 2018):
  https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/fetch_env.py

  compute_reward(achieved_goal, desired_goal, info):
      d = ||achieved_goal - desired_goal||
      sparse: -(d > distance_threshold).astype(float32)   →  0 = success, -1 = fail
      dense:  -d                                           →  always negative, closer = better
      shaped: potential-based Approach → Clamp → Lift shaping (see NOTES_FOR_AI.md
              §5), R_t = Φ(S_t) − Φ(S_{t-1}) plus a +500 success bonus — gives partial
              credit long before a full pick-and-lift, unlike sparse/dense above.

Compatible with:
  • Stable-Baselines3 HerReplayBuffer (HER) — requires GoalEnv dict observation space
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

# Distance threshold from OpenAI FetchPickAndPlace: 5cm (achievable with a 48mm foam ball)
DISTANCE_THRESHOLD = 0.05
# Target lift height: ball must reach 10cm above table to count as "picked up"
TARGET_Z = 0.10

# Added 2026-09-08: a big one-time bonus the instant it genuinely grasps the
# ball (both pads, gripper closed, close enough) AND lifts it AND holds it
# there continuously without dropping it, even before it necessarily reaches
# the exact goal height/position. This is deliberately easier to reach than
# full "success" (which also needs the precise goal position) so the agent
# gets a big, clear signal for the core skill — grasp + lift + don't slip —
# on its own.
STABLE_HOLD_STEPS  = 15     # ~0.3s of unbroken, genuine hold while lifted
STABLE_LIFT_HEIGHT = 0.06   # ball must be at least 6cm off the table (started at ~2.4cm)
STABLE_GRASP_BONUS = 300.0

class SO101PickEnv(gym.Env):
    """
    Gymnasium GoalEnv for SO-101 robot arm picking an orange foam ball.
    Reward is OpenAI-compatible sparse or dense, ready for HER training.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, reward_type="sparse"):
        super().__init__()
        self.render_mode = render_mode
        self.reward_type = reward_type  # "sparse" (OpenAI standard), "dense", or "shaped"

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
        # Matches OpenAI FetchPickAndPlace structure:
        #   "observation"    : robot state (joint pos, joint vel, gripper pos, ball pos, rel vector)
        #   "achieved_goal"  : current ball XYZ position
        #   "desired_goal"   : target XYZ position (ball lifted to TARGET_Z)
        obs_dim = 21  # 6 qpos + 6 qvel + 3 gripper_pos + 3 ball_pos + 3 rel_vec
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-10.0, 10.0, shape=(3,),      dtype=np.float32),
            "desired_goal":  spaces.Box(-10.0, 10.0, shape=(3,),      dtype=np.float32),
        })

        self.distance_threshold = DISTANCE_THRESHOLD

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

        # Goal (set during reset)
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
        """True only if BOTH inner rubber jaw pads touch the ball simultaneously."""
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

    # ── OpenAI-compatible compute_reward ─────────────────────────────────────
    # Exact signature from gymnasium_robotics/envs/fetch/fetch_env.py

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        achieved_goal : ball XYZ (or batch N×3)
        desired_goal  : target XYZ (or batch N×3)
        info          : one dict (a live env.step() call) or an array/list of
                        dicts (stable-baselines3's HerReplayBuffer batches
                        these when relabeling goals for HER).
        Returns:
          sparse: 0.0 if ||ag - dg|| <= threshold, else -1.0
          dense:  -||ag - dg||
          shaped: info["shaping_reward"] (Φ(S_t) − Φ(S_{t-1}), goal-independent,
                  precomputed in step()) plus a +500 success bonus recomputed
                  against whichever desired_goal is passed in, so it stays
                  correct after HER relabels the goal.
        """
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        if self.reward_type == "dense":
            return -d.astype(np.float32)
        if self.reward_type == "shaped":
            shaping    = self._info_field(info, "shaping_reward", 0.0)
            is_held    = self._info_field(info, "is_held", False)
            held_steps = self._info_field(info, "held_in_air_steps", 0)
            success    = (d <= self.distance_threshold) & is_held & (held_steps >= 3)
            jackpot    = np.where(success, 500.0, 0.0)
            return (shaping + jackpot).astype(np.float32)
        raise ValueError(f"Unknown reward_type: {self.reward_type!r}")

    @staticmethod
    def _info_field(info, key, default):
        """Pull `key` out of `info`, whether it's a single dict (a live
        env.step() call) or an array/list of dicts (HER's batched calls)."""
        if isinstance(info, dict):
            return np.asarray(info.get(key, default))
        return np.asarray([d.get(key, default) if isinstance(d, dict) else default for d in info])

    # ── Potential-based reward shaping (Approach → Clamp → Lift) ─────────────
    # See NOTES_FOR_AI.md §5-6. Gives smooth partial credit long before a full
    # pick-and-lift so a sparse-reward agent has something to learn from.

    def _potential(self, dist_to_ball, grip_angle, is_touching, is_held, ball_ascent, hand_ascent):
        GRIP_OPEN = 0.40   # radians — roughly the "jaws open" home angle
        GRIP_SHUT = 0.05   # radians — roughly "jaws clamped shut"

        if is_held:
            # Stage 3 [50 → 200 pts]: genuinely gripping — reward height gained
            ascent = float(np.clip(max(ball_ascent, 0.5 * hand_ascent), 0.0, 1.0))
            return 50.0 + 150.0 * ascent
        if is_touching:
            # Stage 2 [20 → 50 pts]: jaws around the ball — reward closing the grip
            grip_frac = float(np.clip((GRIP_OPEN - grip_angle) / (GRIP_OPEN - GRIP_SHUT), 0.0, 1.0))
            return 20.0 + 30.0 * grip_frac
        # Stage 1 [0 → 20 pts]: still approaching — reward closing distance with open jaws
        approach = 1.0 - float(np.tanh(5.0 * dist_to_ball))
        openness = float(np.clip(grip_angle / GRIP_OPEN, 0.0, 1.0))
        return 20.0 * approach * openness

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
            "desired_goal":  self._goal.copy(),     # target lift XYZ
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step      = 0
        self.held_in_air_steps = 0
        self._stable_bonus_given = False   # resets each episode

        # Randomize ball position on desk
        bx = self.np_random.uniform(0.18, 0.23)
        by = self.np_random.uniform(-0.07, 0.07)
        bz = 0.024
        base_angle = float(np.arctan2(by, bx))

        self.data.qpos[self.ball_qpos_adr     : self.ball_qpos_adr + 3] = [bx, by, bz]
        self.data.qpos[self.ball_qpos_adr + 3 : self.ball_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Goal: ball lifted TARGET_Z above table, same XY as ball (straight up)
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

        # Shaping-reward bookkeeping (potential-based reward, see step())
        self._ball_start_z    = float(bz)
        self._gripper_start_z = float(self._get_pinch_pos()[2])
        self._prev_potential  = self._potential(
            dist_to_ball=float(np.linalg.norm(self.data.xpos[self.ball_body_id] - self._get_pinch_pos())),
            grip_angle=float(self.data.qpos[5]),
            is_touching=self._is_contacting(),
            is_held=False, ball_ascent=0.0, hand_ascent=0.0,
        )

        return self._get_obs(), {}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)

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

        is_touching  = self._is_contacting()
        gripper_pos  = self._get_pinch_pos()
        ball_pos     = self.data.xpos[self.ball_body_id].copy().astype(np.float32)
        dist_to_ball = float(np.linalg.norm(ball_pos - gripper_pos))
        ball_z       = float(ball_pos[2])
        grip_angle   = float(self.data.qpos[5])

        # Sustained hold tracking (anti-flick: both pads + clamped + airborne)
        # Loosened 2026-09-08: was dist<=0.045 / grip<0.20 — too strict a
        # simultaneous window given how long training was stuck at 0% even
        # on curriculum Level 1. See NOTES_FOR_AI.md changelog.
        is_held = bool(is_touching and dist_to_ball <= 0.06 and grip_angle < 0.28)
        if is_held and ball_z > 0.040:
            self.held_in_air_steps += 1
        else:
            self.held_in_air_steps = 0

        # Big one-time bonus: genuinely grasped + lifted + held steady, no slipping.
        stable_grasp_bonus = 0.0
        if (not self._stable_bonus_given
                and is_held
                and ball_z >= STABLE_LIFT_HEIGHT
                and self.held_in_air_steps >= STABLE_HOLD_STEPS):
            stable_grasp_bonus = STABLE_GRASP_BONUS
            self._stable_bonus_given = True

        # ── Potential-based shaping (used only when reward_type == "shaped") ──
        ball_ascent = max(0.0, ball_z - self._ball_start_z)
        hand_ascent = max(0.0, float(gripper_pos[2]) - self._gripper_start_z)
        current_potential    = self._potential(dist_to_ball, grip_angle, is_touching,
                                                is_held, ball_ascent, hand_ascent)
        shaping_reward        = current_potential - self._prev_potential + stable_grasp_bonus
        self._prev_potential  = current_potential

        # ── OpenAI-compatible reward ──────────────────────────────────────────
        achieved_goal = ball_pos
        desired_goal  = self._goal
        info = {
            "dist":              dist_to_ball,
            "ball_z":            ball_z,
            "is_touching":       is_touching,
            "shaping_reward":    shaping_reward,
            "is_held":           is_held,
            "held_in_air_steps": self.held_in_air_steps,
            "grip_angle":        grip_angle,
        }
        reward = float(self.compute_reward(achieved_goal, desired_goal, info))

        # Success: ball within distance_threshold of goal (>=10cm up) AND genuinely held
        success    = bool(
            np.linalg.norm(achieved_goal - desired_goal) <= self.distance_threshold
            and is_held
            and self.held_in_air_steps >= 3
        )
        terminated = success
        truncated  = bool(self.current_step >= self.max_steps)

        obs = self._get_obs()
        info["success"]    = success
        info["is_success"] = success   # SB3 HER reads "is_success" key

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

