#!/usr/bin/env python3
"""
train_vision_ai.py
MPS-Native 3D Pose Estimation Neural Network Training in MuJoCo.
"""

import os
import time
import argparse
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

MEAN_X = 0.25
SCALE_X = 0.10
MEAN_Y = 0.00
SCALE_Y = 0.12

def normalize_target(bx, by):
    nx = (bx - MEAN_X) / SCALE_X
    ny = (by - MEAN_Y) / SCALE_Y
    return nx, ny

def denormalize_target(nx, ny):
    bx = nx * SCALE_X + MEAN_X
    by = ny * SCALE_Y + MEAN_Y
    return bx, by

# ==============================================================================
# 🧠 Vision Neural Network (MPS-Native Coordinate Regressor)
# ==============================================================================
class BallPoseNet(nn.Module):
    def __init__(self):
        super(BallPoseNet, self).__init__()
        self.conv = nn.Sequential(
            # 3 -> 32 (120x160)
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 60x80
            
            # 32 -> 64 (30x40)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 -> 128 (15x20)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 7x10
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # [Norm_X, Norm_Y]
        )

    def forward(self, x):
        feat = self.conv(x)
        out = self.fc(feat)
        return out

    def predict_meters(self, x):
        norm_out = self.forward(x)
        nx = norm_out[:, 0]
        ny = norm_out[:, 1]
        bx = nx * SCALE_X + MEAN_X
        by = ny * SCALE_Y + MEAN_Y
        return torch.stack([bx, by], dim=1)

# ==============================================================================
# 🎮 MuJoCo Synthetic Camera Data Generator
# ==============================================================================
class SyntheticVisionEnv:
    def __init__(self, xml_path="/Users/nam/Arm Bot/so101_mujoco.xml", width=320, height=240):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.width = width
        self.height = height
        
        self.ball_jadr = self.model.joint('ball_joint').qposadr[0]
        self.home_rad = np.deg2rad(np.array([0.0, -50.0, 50.0, 60.0, 0.0, 25.0]))

    def sample_batch(self, batch_size=32):
        images = []
        norm_targets = []
        raw_targets = []
        
        for _ in range(batch_size):
            arm_jitter = np.random.uniform(-0.10, 0.10, size=6)
            self.data.qpos[:6] = self.home_rad + arm_jitter
            
            bx = float(np.random.uniform(0.18, 0.32))
            by = float(np.random.uniform(-0.12, 0.12))
            bz = 0.024
            
            self.data.qpos[self.ball_jadr : self.ball_jadr + 3] = [bx, by, bz]
            mujoco.mj_forward(self.model, self.data)
            
            self.renderer.update_scene(self.data, camera='logi_cam')
            rgb = self.renderer.render()
            
            # Domain Randomization
            img = rgb.astype(np.float32) / 255.0
            brightness = np.random.uniform(0.9, 1.1)
            img = np.clip(img * brightness, 0.0, 1.0)
            
            tensor_img = np.transpose(img, (2, 0, 1))
            images.append(tensor_img)
            
            nx, ny = normalize_target(bx, by)
            norm_targets.append([nx, ny])
            raw_targets.append([bx, by])
            
        return (torch.tensor(np.array(images), dtype=torch.float32), 
                torch.tensor(np.array(norm_targets), dtype=torch.float32),
                np.array(raw_targets))

# ==============================================================================
# 🚀 Training Loop
# ==============================================================================
def train(steps=1500, batch_size=32, lr=1e-3, output_dir="/Users/nam/Arm Bot/outputs/vision_models"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using Device: {device}")
    
    env = SyntheticVisionEnv()
    model = BallPoseNet().to(device)
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    print("\n" + "="*70)
    print("🧠 TRAINING VISION AI (MPS Fast Native Regression)")
    print(f"   • Total Training Steps : {steps:,}")
    print(f"   • Batch Size           : {batch_size}")
    print(f"   • Output Directory     : {output_dir}")
    print("="*70 + "\n")
    
    best_err_mm = 999.0
    start_time = time.time()
    
    for step in range(1, steps + 1):
        model.train()
        images, norm_targets, raw_targets = env.sample_batch(batch_size=batch_size)
        images, norm_targets = images.to(device), norm_targets.to(device)
        
        optimizer.zero_grad()
        norm_preds = model(images)
        loss = criterion(norm_preds, norm_targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            preds_m = model.predict_meters(images).cpu().numpy()
            diff = preds_m - raw_targets
            err_mm = np.mean(np.linalg.norm(diff, axis=1)) * 1000.0
            
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - start_time
            rate = step / max(elapsed, 1e-4)
            print(f"Step {step:5,d}/{steps:,} | Loss: {loss.item():.5f} | Avg Position Error: {err_mm:5.2f} mm | Speed: {rate:5.1f} steps/s")
            
        if err_mm < best_err_mm and step > 30:
            best_err_mm = err_mm
            save_path = os.path.join(output_dir, "best_ball_pose_model.pth")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "error_mm": best_err_mm,
                "loss": loss.item()
            }, save_path)
            
    final_path = os.path.join(output_dir, "final_ball_pose_model.pth")
    torch.save(model.state_dict(), final_path)
    print("\n" + "="*70)
    print(f"🏆 Vision AI Training Complete!")
    print(f"   • Best Position Accuracy: {best_err_mm:.2f} mm ({best_err_mm/10:.2f} cm precision!)")
    print(f"   • Saved to: {os.path.join(output_dir, 'best_ball_pose_model.pth')}")
    print("="*70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    train(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
