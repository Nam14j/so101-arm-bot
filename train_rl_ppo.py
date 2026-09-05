#!/usr/bin/env python3
"""
train_rl_ppo.py — SAC + HER (Hindsight Experience Replay) for SO-101 Ball Pick

Uses the EXACT same training method as OpenAI FetchPickAndPlace:
  • Algorithm  : SAC (off-policy, works great with sparse rewards)
  • Replay     : HerReplayBuffer (relabels failed episodes as successes)
  • Reward     : Sparse -1/0  (OpenAI standard, no reward hacking)
  • Observation: GoalEnv dict {observation, achieved_goal, desired_goal}

Run:
  python train_rl_ppo.py
  python train_rl_ppo.py --timesteps 2000000
  python train_rl_ppo.py --device cpu
"""

import os
import sys
import time
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_env import SO101PickEnv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "rl_models", "so101_sac_her")


def make_env():
    return SO101PickEnv(reward_type="sparse")


# ── Curriculum Callback ────────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """
    Auto-advances curriculum level:
      Level 1 (0–300k)  : Arm starts at ball — just clamp & lift
      Level 2 (300–800k): Arm 4cm above — drop, clamp & lift
      Level 3 (800k+)   : Full workspace from home pose
    """
    def __init__(self, eval_env, verbose=1):
        super().__init__(verbose)
        self.eval_env      = eval_env
        self.current_level = 1

    def _on_training_start(self):
        self._set_level(1)

    def _set_level(self, level):
        self.current_level = level
        self.training_env.env_method("set_curriculum_level", level)
        self.eval_env.env_method("set_curriculum_level", level)
        labels = {
            1: "LEVEL 1 — The Lift (arm already at ball, just clamp & lift)",
            2: "LEVEL 2 — Drop & Clamp (arm 4 cm above ball)",
            3: "LEVEL 3 — Full Workspace (reach from home pose)",
        }
        print("\n" + "=" * 70)
        print(f"🎓 CURRICULUM: {labels[level]}")
        print("=" * 70 + "\n")

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if self.current_level == 1 and t >= 300_000:
            self._set_level(2)
        elif self.current_level == 2 and t >= 800_000:
            self._set_level(3)
        return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int,   default=2_000_000)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--device",    type=str,   default="auto")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("🤖 SO-101 SAC + HER TRAINER  (OpenAI FetchPickAndPlace method)")
    print("=" * 70)
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"  Reward    : Sparse  (0 = success, -1 = fail)")
    print(f"  HER goals : future  (relabels future positions as goals)")
    print("-" * 70)

    env      = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    curriculum_cb = CurriculumCallback(eval_env=eval_env)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=OUTPUT_DIR,
        log_path=OUTPUT_DIR,
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=OUTPUT_DIR,
        name_prefix="so101_sac_her",
    )

    # SAC + HER — MultiInputPolicy handles the GoalEnv dict obs space
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        learning_rate=args.lr,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.98,
        tau=0.05,
        learning_starts=500,   # Must be > max_steps (350) so HER has a full episode to sample from
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        device=args.device,
    )

    print("\n🚀 Starting SAC + HER training...\n")
    start = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[curriculum_cb, eval_cb, checkpoint_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted.")
    finally:
        final_path = os.path.join(OUTPUT_DIR, "final_model.zip")
        model.save(final_path)
        elapsed = time.time() - start
        print(f"\n💾 Saved → {final_path}")
        print(f"⏱️  Elapsed: {elapsed / 60:.1f} minutes")
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
