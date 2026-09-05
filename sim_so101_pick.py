#!/usr/bin/env python3
"""
sim_so101_pick.py — Autonomous SO-101 Ball Picker with Random Ball Relocation in MuJoCo
"""

import os
import sys
import time
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("❌ MuJoCo is not installed in this environment.")
    print("   Please install it with: pip install mujoco")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "so101_mujoco.xml")

def randomize_ball(model, data):
    """Teleports the ball to a randomized reachable spot on the desk far from the gripper."""
    ball_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    grip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
    qpos_adr = model.jnt_qposadr[ball_jnt]
    dof_adr = model.jnt_dofadr[ball_jnt]

    # Sample random coordinates ensuring it never spawns near the gripper
    for _ in range(50):
        x = np.random.uniform(0.18, 0.23)
        y = np.random.uniform(-0.07, 0.07)
        z = 0.022
        dist_to_grip = np.linalg.norm(np.array([x, y, z]) - data.xpos[grip_body])
        if dist_to_grip > 0.12:  # Must be at least 12cm away from gripper
            break

    data.qpos[qpos_adr : qpos_adr + 3] = [x, y, z]
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[dof_adr : dof_adr + 6] = 0.0
    mujoco.mj_forward(model, data)
    print(f"🎲 Ball placed on desk at: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m ({dist_to_grip*100:.1f}cm from gripper)")
    return x, y, z

def solve_reach_angles(bx, by):
    """Calculates joint angles to align with and reach down to the ball."""
    # Pan directly facing the ball
    pan = np.arctan2(by, bx)
    
    # Distance from base
    dist = np.sqrt(bx**2 + by**2)
    
    # Scale reach based on distance
    reach_ratio = np.clip((dist - 0.18) / 0.06, 0.0, 1.0)
    lift_deg = 20.0 + reach_ratio * 12.0
    elbow_deg = -10.0 + reach_ratio * 10.0
    wrist_f_deg = 65.0 - reach_ratio * 8.0
    
    return pan, np.deg2rad(lift_deg), np.deg2rad(elbow_deg), np.deg2rad(wrist_f_deg)

def is_ball_grasped(model, data):
    """Verifies that the ball is physically lifted and held in the gripper."""
    ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    grip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
    
    ball_z = data.xpos[ball_body][2]
    dist_to_grip = np.linalg.norm(data.xpos[ball_body] - data.xpos[grip_body])
    
    # Ball is grasped if it is lifted off the table surface (Z > 5cm) and within gripper reach (< 8cm)
    return ball_z > 0.045 and dist_to_grip < 0.08

def main():
    print("=" * 65)
    print("🤖 SO-101 Verified Ball Picker & Relocation (MuJoCo Physics)")
    print("=" * 65)

    if not os.path.exists(XML_PATH):
        print(f"❌ Model file not found at {XML_PATH}")
        return

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Exact Physical Arm Home Pose from real SO-101 hardware
    home_rad = np.deg2rad(np.array([0.0, -104.0, 94.0, 63.0, 0.0, 5.0]))
    data.qpos[:6] = home_rad
    data.ctrl[:6] = home_rad

    # Spawn first random ball
    bx, by, bz = randomize_ball(model, data)

    print("\n🎮 Controls:")
    print("   • [Space]               : Pause / Resume physics")
    print("   • [Ctrl + Right Click]  : Drag arm or ball with physics force")
    print("   • Ball will ONLY relocate when successfully grasped and transported!")
    print("-" * 65)

    TARGET_FPS = 60.0
    frame_dt = 1.0 / TARGET_FPS
    substeps = max(1, int(frame_dt / model.opt.timestep))

    # Pick State Machine
    state = "APPROACH"
    state_timer = time.time()
    pan_tgt, lift_tgt, elbow_tgt, wf_tgt = solve_reach_angles(bx, by)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            frame_start = time.time()
            elapsed_state = frame_start - state_timer

            # Read live ball position
            ball_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
            cur_ball_pos = data.xpos[ball_body]

            # ─── Autonomous Pick-and-Place Transport State Machine ───
            if state == "APPROACH":
                # Aim at current ball position and open gripper wide
                pan_tgt, lift_tgt, elbow_tgt, wf_tgt = solve_reach_angles(cur_ball_pos[0], cur_ball_pos[1])
                target_ctrl = [pan_tgt, np.deg2rad(-20.0), np.deg2rad(30.0), np.deg2rad(60.0), 0.0, np.deg2rad(45.0)]
                if elapsed_state > 1.2:
                    state = "REACH_DOWN"
                    state_timer = time.time()

            elif state == "REACH_DOWN":
                # Lower gripper directly over the ball
                target_ctrl = [pan_tgt, lift_tgt, elbow_tgt, wf_tgt, 0.0, np.deg2rad(45.0)]
                if elapsed_state > 1.4:
                    state = "GRASP"
                    state_timer = time.time()

            elif state == "GRASP":
                # Clamp gripper tight around ball with rubber holding force
                target_ctrl = [pan_tgt, lift_tgt, elbow_tgt, wf_tgt, 0.0, np.deg2rad(5.0)]
                if elapsed_state > 0.8:
                    state = "LIFT"
                    state_timer = time.time()

            elif state == "LIFT":
                # Lift the arm into the air
                target_ctrl = [pan_tgt, np.deg2rad(-50.0), np.deg2rad(45.0), np.deg2rad(60.0), 0.0, np.deg2rad(5.0)]
                if elapsed_state > 1.5:
                    # VERIFY: Is the ball actually in the gripper?
                    if is_ball_grasped(model, data):
                        print("🎯 Ball successfully grasped in gripper! Transporting to drop zone...")
                        state = "TRANSPORT"
                        state_timer = time.time()
                    else:
                        print("⚠️ Ball not in gripper! Retrying current position...")
                        state = "APPROACH"
                        state_timer = time.time()

            elif state == "TRANSPORT":
                # Swing over to the target delivery drop zone on the side
                drop_pan = np.deg2rad(45.0)  # Move 45 degrees to the left
                target_ctrl = [drop_pan, np.deg2rad(-40.0), np.deg2rad(35.0), np.deg2rad(60.0), 0.0, np.deg2rad(5.0)]
                if elapsed_state > 1.8:
                    state = "PLACE_DOWN"
                    state_timer = time.time()

            elif state == "PLACE_DOWN":
                # Lower the ball smoothly onto the destination spot
                drop_pan = np.deg2rad(45.0)
                target_ctrl = [drop_pan, np.deg2rad(15.0), np.deg2rad(0.0), np.deg2rad(55.0), 0.0, np.deg2rad(5.0)]
                if elapsed_state > 1.4:
                    state = "RELEASE"
                    state_timer = time.time()

            elif state == "RELEASE":
                # Open gripper to release the ball
                drop_pan = np.deg2rad(45.0)
                target_ctrl = [drop_pan, np.deg2rad(15.0), np.deg2rad(0.0), np.deg2rad(55.0), 0.0, np.deg2rad(45.0)]
                if elapsed_state > 0.8:
                    state = "RETRACT"
                    state_timer = time.time()

            elif state == "RETRACT":
                # Raise back up and return towards center
                target_ctrl = [0.0, np.deg2rad(-70.0), np.deg2rad(60.0), np.deg2rad(60.0), 0.0, np.deg2rad(45.0)]
                if elapsed_state > 1.5:
                    # ONLY NOW: Successfully placed the ball -> Teleport to a new random spot!
                    bx, by, bz = randomize_ball(model, data)
                    state = "APPROACH"
                    state_timer = time.time()

            # Smooth target blend (EMA)
            current_ctrl = np.array(data.ctrl[:6])
            smoothed_ctrl = current_ctrl + 0.15 * (np.array(target_ctrl) - current_ctrl)
            data.ctrl[:6] = smoothed_ctrl

            # Step physics
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            viewer.sync()

            sleep_t = max(0.0, frame_dt - (time.time() - frame_start))
            if sleep_t > 0:
                time.sleep(sleep_t)

    print("\n👋 Simulation closed.")

if __name__ == "__main__":
    main()
