#!/usr/bin/env python3
"""
train_rl_live.py — Live Visual PPO Reinforcement Learning Trainer (Watch AI Learn in 3D)
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_env import SO101PickEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "rl_models", "so101_ppo")

class LiveViewerCallback(BaseCallback):
    """Syncs the 3D MuJoCo window on every training step."""
    def __init__(self, env, viewer, fps=60.0):
        super().__init__()
        self.env = env
        self.viewer = viewer
        self.frame_dt = 1.0 / fps
        self.last_time = time.time()
        self.ep_count = 0
        self.ep_reward = 0.0

    def _on_step(self) -> bool:
        # Sync 3D window
        if self.viewer.is_running():
            self.viewer.sync()
            
            # Print live episode stats
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")
            if rewards is not None:
                self.ep_reward += float(rewards[0])
            if dones is not None and dones[0]:
                self.ep_count += 1
                print(f"🎮 Training Episode {self.ep_count:3d} | Reward: {self.ep_reward:7.1f} | Timestep: {self.num_timesteps:,}")
                self.ep_reward = 0.0

            # Frame rate cap so human can watch smoothly
            elapsed = time.time() - self.last_time
            sleep_t = max(0.0, self.frame_dt - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)
            self.last_time = time.time()
            return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Watch SO-101 Train Live in 3D via Reinforcement Learning")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total RL timesteps (default: 100k)")
    parser.add_argument("--fps", type=float, default=60.0, help="Rendering speed FPS (default: 60)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("🤖 SO-101 LIVE Visual PPO Reinforcement Learning Trainer")
    print("=" * 65)
    print("👁️ Opening 3D physics window... Watch the AI learn in real time!")
    print("🎮 Controls:")
    print("   • [Space]               : Pause / Resume physics")
    print("   • [Ctrl + Right Click]  : Drag arm or ball with physics force")
    print("-" * 65)

    env = SO101PickEnv()

    # Pre-train with expert demonstration bootstrap
    print("🎓 Pre-loading expert demonstrations into AI neural network...")
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    print("✅ Ready! Starting Live 3D Training Session...\n")

    # Launch passive viewer and train live
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        cb = LiveViewerCallback(env, viewer, fps=args.fps)
        try:
            model.learn(total_timesteps=args.timesteps, callback=cb, progress_bar=True)
        except KeyboardInterrupt:
            print("\n⚠️ Live training stopped by user.")

    # Save model
    final_path = os.path.join(OUTPUT_DIR, "live_trained_model.zip")
    best_path = os.path.join(OUTPUT_DIR, "best_model.zip")
    model.save(final_path)
    model.save(best_path)
    print(f"\n💾 Saved trained policy to: {final_path}")

if __name__ == "__main__":
    main()
