#!/usr/bin/env python3
"""
eval_rl.py — Interactive 3D MuJoCo Visualizer for Trained SO-101 Policies (SAC + HER or PPO)
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco.viewer
from stable_baselines3 import SAC, PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from so101_env import SO101PickEnv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HER_MODEL_DIR = os.path.join(BASE_DIR, "outputs", "rl_models", "so101_sac_her")
PPO_MODEL_DIR = os.path.join(BASE_DIR, "outputs", "rl_models", "so101_ppo")

def find_latest_model():
    candidates = [
        os.path.join(HER_MODEL_DIR, "best_model.zip"),
        os.path.join(HER_MODEL_DIR, "final_model.zip"),
        os.path.join(PPO_MODEL_DIR, "best_model.zip"),
        os.path.join(PPO_MODEL_DIR, "final_model.zip"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

def load_policy(model_path, env):
    try:
        model = SAC.load(model_path, env=env)
        algo = "SAC (HER)"
        return model, algo
    except Exception:
        model = PPO.load(model_path, env=env)
        algo = "PPO"
        return model, algo

def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize trained SO-101 policy in 3D MuJoCo")
    parser.add_argument("--model_path", type=str, default="", help="Path to .zip model (default: auto-detect latest)")
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3], help="Curriculum level: 1 (Lift), 2 (Drop), 3 (Full Workspace)")
    args = parser.parse_args()

    model_path = args.model_path
    if not model_path:
        model_path = find_latest_model()
        if not model_path:
            print("❌ No trained RL model found in outputs/rl_models.")
            print("   Please train one with: python train_rl_ppo.py")
            return

    print("=" * 65)
    print("🤖 SO-101 Trained RL Policy 3D MuJoCo Visualizer")
    print("=" * 65)
    print(f"📂 Loading Model : {model_path}")
    print(f"🎓 Curriculum    : Level {args.level}")

    env = SO101PickEnv()
    env.set_curriculum_level(args.level)

    model, algo = load_policy(model_path, env)
    print(f"✅ Algorithm     : {algo}")

    print("\n🎮 Controls in 3D Viewer:")
    print("   • [Space]               : Pause / Resume physics")
    print("   • [Ctrl + Right Click]  : Drag arm or ball with physics force")
    print("   • Press [Esc] or close window to exit")
    print("-" * 65)

    TARGET_FPS = 50.0
    frame_dt = 1.0 / TARGET_FPS

    obs, info = env.reset()
    episodes_completed = 0
    ep_reward = 0.0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            frame_start = time.time()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            viewer.sync()

            if terminated or truncated:
                episodes_completed += 1
                status = "🏆 SUCCESS (Ball Lifted!)" if info.get("success") else "⏱️ TIMEOUT"
                grip_deg = np.rad2deg(env.data.qpos[5])
                dist_cm = info.get("dist", 0.0) * 100.0
                ball_z_cm = info.get("ball_z", 0.0) * 100.0
                print(f"Episode {episodes_completed:2d}: {status} | Dist: {dist_cm:4.1f}cm | Ball Z: {ball_z_cm:4.1f}cm | Grip: {grip_deg:4.1f}° | Reward: {ep_reward:.1f}")
                ep_reward = 0.0
                obs, info = env.reset()
                time.sleep(0.3)

            elapsed = time.time() - frame_start
            sleep_t = max(0.0, frame_dt - elapsed)
            if sleep_t > 0:
                time.sleep(sleep_t)

    print("\n👋 3D Visualizer closed.")

if __name__ == "__main__":
    main()


