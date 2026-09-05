import time
import torch

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

config = SO101LeaderConfig(
    port="/dev/tty.usbmodem5AE60529841",  # Your leader port
    id="my_leader_arm",                  # The name you gave the leader during calibration
    use_degrees=True                      # True outputs angles in degrees
)

config_follower = SO101FollowerConfig(
    port="/dev/tty.usbmodem5AE60587831",  # Your follower port
    id="my_follower_arm",                # The name you gave the follower during calibration
    use_degrees=True                      # True outputs angles in degrees
)

# 2. Instantiate and connect to the Leader and Follower arms
leader = SO101Leader(config)
leader.connect()  # Automatically loads your local calibration file!

follow = SO101Follower(config_follower)
follow.connect()  # Automatically loads your local calibration file!

print("\nReading joint positions in degrees (Ctrl+C to stop):")

# Initialize an empty list to record the movement path
positions_movement = []

try:
    while True:
        positions = leader.get_action()
        positions_movement.append(positions.copy())
        
        formatted = ", ".join(f"{joint.split('.')[0]}: {angle:.1f}°" for joint, angle in positions.items())
        print(f"\r{formatted}", end="", flush=True)
        
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\nStopped reading positions.")
    print(f"Recorded {len(positions_movement)} steps of movement.")
    
    if len(positions_movement) == 0:
        print("No movements recorded. Exiting.")
    else:
        print("\nMoving follower safely to the initial recorded position...")
        follow.send_action(positions_movement[0])
        time.sleep(2.0) 
        
        print("Starting playback loop on follower arm... (Ctrl+C to exit playback)")
        try:
            while True:
                for target_position in positions_movement:
                    follow.send_action(target_position)
                    time.sleep(0.1) 
        except KeyboardInterrupt:
            print("\nPlayback loop terminated by user.")
    
finally:
    print("Disconnecting devices safely...")
    leader.disconnect()
    follow.disconnect()
