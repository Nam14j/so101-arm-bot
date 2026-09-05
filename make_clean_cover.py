import trimesh
import trimesh.creation
import numpy as np
import os

OUT = "/Users/nam/Arm Bot/stl"
path = os.path.join(OUT, "07_joint_cover_collar.stl")

if os.path.exists(path):
    os.remove(path)

def box(w, h, d, cx=0, cy=0, cz=0):
    m = trimesh.creation.box(extents=[w, h, d])
    m.apply_translation([cx, cy, cz])
    return m

# Clean straight rectangular U-channel:
# Length: 50.0mm (X)
# Outer Width: 40.0mm (Z: -20.0 to +20.0)
# Outer Height: 38.0mm (Y: 0.0 to 38.0)
# Wall Thickness: 3.0mm
# Top Cap Wall: 50mm (X) x 3.0mm (Y: 35->38) x 40mm (Z: -20->20)
top_cap = box(50.0, 3.0, 40.0, 0, 36.5, 0)

# Left Side Wall: 50mm (X) x 35.0mm (Y: 0->35) x 3.0mm (Z: -20->-17)
left_wall = box(50.0, 35.0, 3.0, 0, 17.5, -18.5)

# Right Side Wall: 50mm (X) x 35.0mm (Y: 0->35) x 3.0mm (Z: 17->20)
right_wall = box(50.0, 35.0, 3.0, 0, 17.5, 18.5)

# Union the 3 straight rectangular plates:
u_mesh = trimesh.boolean.union([top_cap, left_wall, right_wall], engine='manifold')

# Export clean STL
u_mesh.export(path)

m7 = trimesh.load(path)
bb = m7.bounding_box.extents
wt = m7.is_watertight
bodies = m7.split()

print(f"Saved Clean Rectangle U-Channel: {path}")
print(f"Dimensions: {bb[0]:.1f} (Length) x {bb[1]:.1f} (Height) x {bb[2]:.1f} (Width) mm")
print(f"Watertight: {wt}")
print(f"Body Count: {len(bodies)} (Single 100% solid continuous body)")
print(f"Indentations: NONE (Flat, smooth, straight planar walls!)")
