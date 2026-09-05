# SO-101 Arm Bot 🤖

Autonomous ball pick-and-place system for the **SO-101 6-DOF Robotic Arm** using **MuJoCo physical simulation, Action Chunking with Transformers (ACT), Computer Vision, and Reinforcement Learning (SAC + HER)**.

---

## 🌟 Key Features

1. **High-Fidelity MuJoCo 3D Physics Simulation**:
   - Accurately modeled Feetech STS3215 servo kinematics and torque limits.
   - Contact modeling for inner rubber finger pads and orange foam ball.
   - Off-screen and passive interactive 3D viewers (`mjpython`).

2. **Action Chunking with Transformers (ACT)**:
   - Trained on 50 real-world teleoperation demonstration episodes.
   - Full end-to-end trajectory execution.

3. **Computer Vision Tracker**:
   - OpenCV HSV color segmentation + perspective homography projection.
   - Dynamic real-time millimeter tracking of ball 3D coordinates.

4. **Reinforcement Learning (SAC + HER)**:
   - OpenAI `FetchPickAndPlace` compatible GoalEnv environment (`so101_env.py`).
   - Reverse Curriculum Learning: Level 1 (Lift) $\to$ Level 2 (Drop) $\to$ Level 3 (Full Workspace).

---

## 🚀 Quick Start

### 1. Requirements & Setup
```bash
# Clone the repository
git clone https://github.com/Nam14j/so101-arm-bot.git
cd so101-arm-bot

# Activate environment (e.g. conda)
conda activate lerobot
pip install mujoco gymnasium stable-baselines3 opencv-python
```

### 2. Download Pretrained ACT Model
Download `best_model.pt` from the [Releases](https://github.com/Nam14j/so101-arm-bot/releases/tag/v1.0.0) tab and place it in `saved_models/`:
```bash
mkdir -p saved_models
gh release download v1.0.0 --pattern "best_model.pt" --dir saved_models/
```

### 3. Run Simulation (`./sim`)
```bash
# Run ACT model pick-and-place
./sim act

# Run Reinforcement Learning (SAC + HER) policy
./sim rl

# Run Vision AI ball tracking in MuJoCo
./sim vision-ai

# Run Automated IK state machine
./sim ik
```

---

## 📁 Repository Structure

```
├── sim                    # One-command simulation launcher
├── so101_env.py           # Gymnasium GoalEnv environment for MuJoCo
├── so101_mujoco.xml       # MuJoCo 3D scene definition
├── so101_new_calib.urdf   # Robot kinematics model
├── train_rl_ppo.py        # SAC + HER RL training script
├── eval_rl.py             # 3D interactive viewer for RL policy
├── sim_so101_act.py       # ACT neural network policy runner
├── vision_detector.py     # Live camera ball tracker
├── stl/                   # 3D printable hardware STL files
└── web/                   # Teleoperation web interface
```

---

## 📜 License
MIT License
