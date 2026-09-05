import time
import sys
import numpy as np
from ikpy.chain import Chain
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# Control the speed: higher duration (seconds) = slower motion
MOVE_DURATION = 3.0  # 3 seconds to complete the movement
CONTROL_LOOP_HZ = 50  # Number of commands sent per second

# Tells the program to exit.
class QuitRequested(Exception):
    pass

def get_float_input(prompt, default_val=None):
    """Safely gets a float input from the user with support for a default value."""
    while True:
        try:
            default_str = f" [{default_val}]" if default_val is not None else ""
            user_input = input(f"{prompt}{default_str}: ").strip()
            
            if user_input.lower() == 'q':
                raise QuitRequested()
                
            if not user_input and default_val is not None:
                return default_val
                
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number or 'q' to quit.")

# Moves the arm slowly to the target position over a specified duration.
def move_smoothly(follow, start_state, target_state, duration=3.0):
    steps = int(duration * CONTROL_LOOP_HZ)
    sleep_interval = 1.0 / CONTROL_LOOP_HZ
    
    # Checks if the URDF has any joints not on the robot and filters them out.
    filtered_target = {}
    for joint_name, target_val in target_state.items():
        if joint_name in start_state:
            filtered_target[joint_name] = target_val

    for step in range(1, steps + 1):
        # Calculates how far along the movement we are.
        t = step / steps
        current_action = {}
        for joint_name, target_val in filtered_target.items():
            start_val = start_state[joint_name]
            # Calculates where the motor should be at each step.
            current_action[joint_name] = start_val + t * (target_val - start_val)
        # Sends the new step to the arm and waits for the next step.
        follow.send_action(current_action)
        time.sleep(sleep_interval)

def main():
    # 1. Load the URDF kinematic chain
    urdf_file = "so101_new_calib.urdf"
    try:
        my_chain = Chain.from_urdf_file(urdf_file)
        print("Successfully loaded the SO-101 URDF.")
    except Exception as e:
        print(f"Error loading URDF: {e}")
        return

    # 2. Configure the Follower Arm
    config_follower = SO101FollowerConfig(
        port="/dev/tty.usbmodem5AE60587831",  # Your follower port
        id="my_follower_arm",                # The name you gave the follower during calibration
        use_degrees=True                      # True outputs angles in degrees
    )
    
    print("\nConnecting to the SO-101 follower arm...")
    follow = SO101Follower(config_follower)
    
    try:
        follow.connect()
        print("Connected successfully!")
        
        # Record the startup position
        initial_pose = follow.get_observation()
        print("Startup position recorded.")
        
        while True:
            print("\n" + "="*50)
            print("Enter target coordinates in METERS (e.g., 0.18 = 18cm).")
            print("Type 'q' at any prompt to return home and quit.")
            print("="*50)
            
            # Read current state to know where we are starting from for the next move
            current_state = follow.get_observation()
            
            # Get target coordinates interactively (can raise QuitRequested)
            x = get_float_input("Enter X (forward/back)", default_val=0.18)
            y = get_float_input("Enter Y (left/right)", default_val=0.0)
            z = get_float_input("Enter Z (up/down)", default_val=0.12)
            
            target_xyz = [x, y, z]
            print(f"\nSolving IK for target: X = {x:.3f}m, Y = {y:.3f}m, Z = {z:.3f}m")
            
            # 3. Compute Inverse Kinematics
            joint_angles_rad = my_chain.inverse_kinematics(target_xyz)
            
            # Convert active joint angles to degrees
            target_angles_deg = {}
            for i, angle in enumerate(joint_angles_rad):
                link = my_chain.links[i]
                if my_chain.active_links_mask[i] and link.name not in ["Base link", "gripper_frame_joint"]:
                    target_angles_deg[f"{link.name}.pos"] = float(np.degrees(angle))
            
            # Checks if the math solver mistakenly computes an angle for a joint that isn't plugged in or shouldn't move
            target_action = {}
            for joint_name, deg_val in target_angles_deg.items():
                if joint_name in current_state:
                    target_action[joint_name] = deg_val
            
            # Preserve gripper position
            if "gripper.pos" in current_state:
                target_action["gripper.pos"] = current_state["gripper.pos"]
            else:
                target_action["gripper.pos"] = 0.0

            # 4. Move the physical arm to target positions
            print(f"Moving arm slowly to target...")
            move_smoothly(follow, current_state, target_action, MOVE_DURATION)
            print("Movement completed!")
            
    # It reads the arm's current position, and calls move_smoothly to drive it back to the initial_pose before safely disconnecting the hardware whenever q or ctrl c is pressed.
    except (QuitRequested, KeyboardInterrupt):
        print("\nReturning arm to its initial startup position...")
        try:
            # Read the final position before starting return trip
            final_state = follow.get_observation()
            move_smoothly(follow, final_state, initial_pose, MOVE_DURATION)
            print("Returned to startup position safely.")
        except Exception as e:
            print(f"Could not return to initial position: {e}")
            
    except Exception as e:
        print(f"\nAn error occurred during communication: {e}")
        
    finally:
        print("\nDisconnecting device safely...")
        try:
            follow.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()