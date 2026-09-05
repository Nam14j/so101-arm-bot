#!/usr/bin/env python3
"""
sim_so101_act.py — Autonomous SO-101 Ball Picker in MuJoCo Physics with 50k ACT Policy
"""

import os
import sys
import time
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("❌ MuJoCo is not installed in this environment.")
    print("   Please install it with: pip install mujoco")
    sys.exit(1)

# Import ACT Policy architecture
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_act_worker import SimpleACTPolicy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "so101_mujoco.xml")
MODEL_PATH = os.path.join(BASE_DIR, "outputs/train/pick_ball_so101_act_20260829_214131/best_model.pt")

# Joint list in degrees for ACT model
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

def main():
    print("=" * 65)
    print("🤖 SO-101 Autonomous ACT Policy (50,000 Steps) in MuJoCo")
    print("=" * 65)

    if not os.path.exists(XML_PATH):
        print(f"❌ Model file not found at {XML_PATH}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ ACT model checkpoint not found at {MODEL_PATH}")
        return

    # 1. Load MuJoCo Model & Physics Data
    print(f"📂 Loading Scene: {XML_PATH}...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # 2. Load 50k ACT PyTorch Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🧠 Loading 50k ACT policy from {MODEL_PATH} on {device}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    chunk_size = checkpoint.get("args", {}).get("chunk_size", 30)

    stats = checkpoint.get("stats", {})
    qpos_mean = np.array(stats.get("qpos_mean", [0.0]*6), dtype=np.float32)
    qpos_std = np.array(stats.get("qpos_std", [1.0]*6), dtype=np.float32)
    action_mean = np.array(stats.get("action_mean", [0.0]*6), dtype=np.float32)
    action_std = np.array(stats.get("action_std", [1.0]*6), dtype=np.float32)

    policy = SimpleACTPolicy(action_dim=6, state_dim=6, chunk_size=chunk_size).to(device)
    policy.load_state_dict(state_dict)
    policy.eval()
    print(f"✅ ACT Policy loaded (Training Step: {checkpoint.get('step', 50000)}, chunk_size={chunk_size}).")

    # Image transform for ResNet backbone
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Offscreen Camera Renderer for AI Vision
    renderer = mujoco.Renderer(model, height=224, width=224)

    # Ready pose: exact starting pose from training demonstrations
    ready_deg = np.array([-0.75, -104.0, 91.0, 74.0, 5.0, 9.0], dtype=np.float32)
    ready_rad = np.deg2rad(ready_deg)
    data.qpos[:6] = ready_rad
    data.ctrl[:6] = ready_rad
    mujoco.mj_forward(model, data)

    print("\n🎮 Launching 3D Simulation...")
    print("   • Watching simulated overhead camera and running ACT neural network @ 30Hz")
    print("   • Press [Space] in the viewer to pause/resume.")
    print("-" * 65)

    # Speed multiplier for swift, crisp robot motion
    SPEED_MULTIPLIER = 1.35
    TRAJ_FPS = 30.0 * SPEED_MULTIPLIER
    ENSEMBLE_EXP_WEIGHT = 0.03
    active_chunks = []
    last_infer_time = 0.0
    TARGET_FPS = 60.0
    frame_dt = 1.0 / TARGET_FPS
    substeps_per_frame = max(1, int(frame_dt / model.opt.timestep))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.time()
            now = frame_start

            # Run ACT Neural Network Inference every ~33ms (30 FPS)
            if now - last_infer_time >= 0.033:
                last_infer_time = now

                # 1. Render camera image from MuJoCo overhead camera
                renderer.update_scene(data, camera="logi_cam")
                cam_rgb = renderer.render()

                # Convert RGB numpy to normalized PyTorch tensor
                pil_img = Image.fromarray(cam_rgb)
                img_tensor = transform(pil_img).unsqueeze(0).to(device)

                # 2. Read current robot joint angles in DEGREES
                curr_deg = np.rad2deg(data.qpos[:6])
                norm_q = (curr_deg - qpos_mean) / qpos_std
                qpos_tensor = torch.tensor([norm_q], dtype=torch.float32).to(device)

                # 3. Predict action chunk
                with torch.no_grad():
                    pred_norm_chunk = policy(img_tensor, qpos_tensor)[0].cpu().numpy()

                # De-normalize predicted degrees
                pred_chunk_deg = pred_norm_chunk * action_std + action_mean

                # Append to temporal ensemble buffer
                active_chunks.append({
                    "chunk": pred_chunk_deg,
                    "start_time": now
                })
                while active_chunks and (now - active_chunks[0]["start_time"] > 1.2):
                    active_chunks.pop(0)

            # High-speed motor blending from active chunks
            if active_chunks:
                weights_sum = 0.0
                weighted_action_sum = np.zeros(6, dtype=np.float32)

                for item in active_chunks:
                    c = item["chunk"]
                    t_start = item["start_time"]
                    age = max(0.0, now - t_start)
                    step_idx = int(age * TRAJ_FPS)

                    if step_idx < len(c):
                        w = np.exp(-ENSEMBLE_EXP_WEIGHT * step_idx)
                        weighted_action_sum += w * c[step_idx]
                        weights_sum += w

                if weights_sum > 0:
                    target_deg = weighted_action_sum / weights_sum
                else:
                    target_deg = active_chunks[-1]["chunk"][-1]

                # Convert target degrees to radians for MuJoCo position actuators
                target_rad = np.deg2rad(target_deg)
                data.ctrl[:6] = target_rad

            # Advance physics substeps per frame for accurate real-time speed
            for _ in range(substeps_per_frame):
                mujoco.mj_step(model, data)

            # Sync GUI viewer
            viewer.sync()

            # Maintain accurate 60 FPS wall-clock timing
            elapsed = time.time() - frame_start
            sleep_time = max(0.0, frame_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n👋 Simulation closed.")

if __name__ == "__main__":
    main()
