import numpy as np
from ikpy.chain import Chain

# Load the URDF file
urdf_file = "so101_new_calib.urdf"
try:
    my_chain = Chain.from_urdf_file(urdf_file)
    print("Successfully loaded the SO-101 URDF!")
    
    # Figures out which motors it can use.
    print("\nKinematic Links:")
    for i, link in enumerate(my_chain.links):
        is_active = my_chain.active_links_mask[i]
        print(f"Link {i}: {link.name} (Active: {is_active})")
        
    # Define a test coordinates (x, y, z) in meters relative to base
    target_position = [3, 5, 0]
    
    # Finds how much you need to move the joints to get to the target position
    joint_angles = my_chain.inverse_kinematics(target_position)
    
    # Prints the angles the joints need to move to in order to reach the target position
    print("\nSolved Joint Angles (Radians & Degrees):")
    for i, angle in enumerate(joint_angles):
        link = my_chain.links[i]
        if my_chain.active_links_mask[i]:
            print(f"  {link.name}: {angle:.4f} rad ({np.degrees(angle):.2f}°)")
            
    # Verify if you got to the right place using forward kinematics. 
    # Basically doing the math backwards (using the calculated angles) 
    # to see if the arm's final position matches our original target coordinates.
    real_frame = my_chain.forward_kinematics(joint_angles)
    computed_position = real_frame[:3, 3]
    print(f"\nTarget Position: {target_position}")
    print(f"Computed Position: {[round(c, 4) for c in computed_position]}")
    
except Exception as e:
    print(f"Error loading or processing URDF: {e}")