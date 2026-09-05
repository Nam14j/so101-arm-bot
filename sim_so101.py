#!/usr/bin/env python3
"""
sim_so101.py — Interactive MuJoCo Simulation for SO-101 Robot Arm
"""

import os
import sys
import time

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("❌ MuJoCo is not installed in this environment.")
    print("   Please install it with: pip install mujoco")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "so101_mujoco.xml")

def main():
    print("=" * 60)
    print("🤖 SO-101 MuJoCo Physics Simulation")
    print("=" * 60)

    if not os.path.exists(XML_PATH):
        print(f"❌ Model file not found at {XML_PATH}")
        return

    print(f"📂 Loading Actuated SO-101 Model: {XML_PATH}...")
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ Failed to load model in MuJoCo: {e}")
        return

    print(f"✅ Successfully loaded SO-101 ({model.nq} generalized coordinates, {model.nu} actuators).")
    print("\n🎮 Controls:")
    print("   • [Space]               : Pause / Resume physics")
    print("   • [Ctrl + Right Click]  : Click and drag any link with physics forces")
    print("   • [Left Click + Drag]   : Rotate 3D camera")
    print("   • [Right Click + Drag]  : Zoom camera")
    print("   • [Double Click Joint]  : Select joint for telemetry")
    print("-" * 60)

    # Launch the interactive GUI viewer
    print("🚀 Launching interactive 3D viewer window...")
    mujoco.viewer.launch(model, data)
    print("\n👋 Simulation closed.")

if __name__ == "__main__":
    main()
