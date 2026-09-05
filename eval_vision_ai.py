#!/usr/bin/env python3
"""
eval_vision_ai.py
Evaluates and visually demonstrates the trained Vision AI Neural Network in MuJoCo.
"""

import os
import time
import mujoco
import numpy as np
import torch
import cv2
from train_vision_ai import BallPoseNet, SyntheticVisionEnv

def evaluate():
    model_path = "/Users/nam/Arm Bot/outputs/vision_models/best_ball_pose_model.pth"
    if not os.path.exists(model_path):
        model_path = "/Users/nam/Arm Bot/outputs/vision_models/final_ball_pose_model.pth"
        
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at {model_path}. Train the model first with train_vision_ai.py!")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = BallPoseNet().to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"✅ Loaded Vision AI Model from: {model_path}")
    
    env = SyntheticVisionEnv(width=640, height=480)
    
    print("\n" + "="*70)
    print("🎯 VISION AI LIVE NEURAL NETWORK EVALUATOR")
    print("   • Green Circle = Ground Truth Ball (MuJoCo Physics)")
    print("   • Cyan Crosshair = AI Neural Network Prediction from Raw Image")
    print("   • Press 'SPACE' for next random position, or 'q' to quit")
    print("="*70 + "\n")
    
    for i in range(1, 50):
        # 1. Randomize ball
        bx = float(np.random.uniform(0.18, 0.32))
        by = float(np.random.uniform(-0.12, 0.12))
        bz = 0.024
        
        env.data.qpos[env.ball_jadr : env.ball_jadr + 3] = [bx, by, bz]
        mujoco.mj_forward(env.model, env.data)
        
        # 2. Render from overhead camera
        env.renderer.update_scene(env.data, camera='logi_cam')
        rgb = env.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # 3. Model Prediction
        img = (rgb.astype(np.float32) / 255.0)
        tensor_img = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model.predict_meters(tensor_img).cpu().numpy()[0]
            
        pred_x, pred_y = pred[0], pred[1]
        err_mm = np.linalg.norm([pred_x - bx, pred_y - by]) * 1000.0
        
        print(f"Test #{i:2d} | True: [X={bx*100:+.1f}cm, Y={by*100:+.1f}cm] | AI: [X={pred_x*100:+.1f}cm, Y={pred_y*100:+.1f}cm] | Error: {err_mm:5.2f} mm")
        
        # Overlay on display frame
        cv2.putText(bgr, f"Ground Truth: X={bx*100:+.1f}cm  Y={by*100:+.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"AI Predicted: X={pred_x*100:+.1f}cm  Y={pred_y*100:+.1f}cm", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"Accuracy Error: {err_mm:.1f} mm", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255) if err_mm > 15 else (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Vision AI - Neural Network Evaluation", bgr)
        key = cv2.waitKey(1200) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluate()
