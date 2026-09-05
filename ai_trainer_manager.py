#!/usr/bin/env python3
"""
ai_trainer_manager.py — AI Dataset Recording, ACT Model Training, and Autonomous Inference Engine
"""

import os
import sys
import time
import json
import threading
import subprocess
from collections import deque
import numpy as np
from PIL import Image
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "train")

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class DatasetManager:
    @staticmethod
    def list_datasets():
        datasets = []
        for name in sorted(os.listdir(DATASETS_DIR)):
            dir_path = os.path.join(DATASETS_DIR, name)
            if os.path.isdir(dir_path):
                ep_files = [f for f in os.listdir(dir_path) if f.startswith("episode_") and f.endswith(".npz")]
                meta_path = os.path.join(dir_path, "metadata.json")
                meta = {}
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                datasets.append({
                    "name": name,
                    "episodes": len(ep_files),
                    "task": meta.get("task", "Pick up the ball from the desk"),
                    "created_at": meta.get("created_at", "")
                })
        return datasets

    @staticmethod
    def get_dataset_info(dataset_name):
        ds_dir = os.path.join(DATASETS_DIR, dataset_name)
        if not os.path.exists(ds_dir):
            return {"name": dataset_name, "episodes": [], "task": "Pick up the ball from the desk", "count": 0}

        meta_path = os.path.join(ds_dir, "metadata.json")
        task_desc = "Pick up the ball from the desk"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    task_desc = json.load(f).get("task", task_desc)
            except Exception:
                pass

        ep_files = sorted([f for f in os.listdir(ds_dir) if f.startswith("episode_") and f.endswith(".npz")])
        episodes = []
        for f in ep_files:
            try:
                ep_path = os.path.join(ds_dir, f)
                data = np.load(ep_path, allow_pickle=True)
                ep_idx = int(f.replace("episode_", "").replace(".npz", ""))
                thumb_file = f.replace(".npz", "_thumb.jpg")
                thumb_exists = os.path.exists(os.path.join(ds_dir, thumb_file))
                episodes.append({
                    "index": ep_idx,
                    "filename": f,
                    "steps": len(data["qpos"]),
                    "duration_s": round(float(data["timestamps"][-1] - data["timestamps"][0]), 1) if len(data["timestamps"]) > 1 else 0.0,
                    "thumb": f"/api/ai/dataset/thumb?dataset={dataset_name}&file={thumb_file}" if thumb_exists else None
                })
            except Exception as e:
                print(f"[Dataset] Error reading {f}: {e}")

        return {
            "name": dataset_name,
            "task": task_desc,
            "count": len(episodes),
            "episodes": episodes
        }

    @staticmethod
    def save_episode(dataset_name, task_desc, qpos_list, actions_list, images_list, timestamps_list):
        ds_dir = os.path.join(DATASETS_DIR, dataset_name)
        os.makedirs(ds_dir, exist_ok=True)

        meta_path = os.path.join(ds_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump({
                "name": dataset_name,
                "task": task_desc,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)

        existing_indices = []
        for f in os.listdir(ds_dir):
            if f.startswith("episode_") and f.endswith(".npz"):
                try:
                    num_str = f.replace("episode_", "").replace(".npz", "")
                    existing_indices.append(int(num_str))
                except ValueError:
                    pass
        ep_idx = (max(existing_indices) + 1) if existing_indices else 0
        
        ep_filename = f"episode_{ep_idx:04d}.npz"
        ep_path = os.path.join(ds_dir, ep_filename)

        # Convert to numpy arrays
        qpos_arr = np.array(qpos_list, dtype=np.float32)
        actions_arr = np.array(actions_list, dtype=np.float32)
        timestamps_arr = np.array(timestamps_list, dtype=np.float64)
        
        # Save npz
        np.savez_compressed(
            ep_path,
            qpos=qpos_arr,
            actions=actions_arr,
            images=images_list,
            timestamps=timestamps_arr
        )

        # Save thumbnail from middle frame
        if images_list and len(images_list) > 0:
            mid_idx = len(images_list) // 2
            mid_img_bytes = images_list[mid_idx]
            if isinstance(mid_img_bytes, bytes):
                thumb_path = os.path.join(ds_dir, f"episode_{ep_idx:04d}_thumb.jpg")
                try:
                    img = Image.open(io.BytesIO(mid_img_bytes)).convert("RGB")
                    img.thumbnail((320, 180))
                    img.save(thumb_path, "JPEG", quality=80)
                except Exception as e:
                    print(f"[Dataset] Thumb error: {e}")

        print(f"[Dataset] Saved episode {ep_idx} ({len(qpos_list)} steps) to {ep_path}")
        return ep_idx

    @staticmethod
    def delete_episode(dataset_name, ep_idx):
        ds_dir = os.path.join(DATASETS_DIR, dataset_name)
        if not os.path.exists(ds_dir):
            return False

        ep_filename = f"episode_{ep_idx:04d}.npz"
        ep_path = os.path.join(ds_dir, ep_filename)
        thumb_path = os.path.join(ds_dir, f"episode_{ep_idx:04d}_thumb.jpg")
        
        deleted = False
        if os.path.exists(ep_path):
            os.remove(ep_path)
            deleted = True
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

        # Renumber remaining episodes to ensure contiguous indexing 0000, 0001, 0002...
        ep_files = sorted([f for f in os.listdir(ds_dir) if f.startswith("episode_") and f.endswith(".npz")])
        for new_idx, old_f in enumerate(ep_files):
            old_npz = os.path.join(ds_dir, old_f)
            new_npz = os.path.join(ds_dir, f"episode_{new_idx:04d}.npz")
            if old_npz != new_npz:
                os.rename(old_npz, new_npz)

            old_thumb = os.path.join(ds_dir, old_f.replace(".npz", "_thumb.jpg"))
            new_thumb = os.path.join(ds_dir, f"episode_{new_idx:04d}_thumb.jpg")
            if os.path.exists(old_thumb) and old_thumb != new_thumb:
                os.rename(old_thumb, new_thumb)

        # Update metadata.json
        meta_path = os.path.join(ds_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                meta["total_episodes"] = len(ep_files)
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        print(f"[Dataset] Deleted episode {ep_idx} from {dataset_name}. {len(ep_files)} episodes remaining.")
        return deleted


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class TrainingManager:
    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.state = {
            "is_training": False,
            "dataset": "",
            "policy_type": "act",
            "step": 0,
            "total_steps": 25000,
            "loss": 0.0,
            "lr": 0.0,
            "fps": 0.0,
            "eta": "--:--:--",
            "logs": deque(maxlen=200),
            "output_dir": "",
            "error": None,
            "start_time": 0.0
        }

    def start_training(self, dataset_name, steps=25000, batch_size=8, lr=2e-4, device="mps", chunk_size=30):
        with self.lock:
            if self.state["is_training"]:
                return {"status": "error", "message": "Training already in progress"}

            ds_dir = os.path.join(DATASETS_DIR, dataset_name)
            if not os.path.exists(ds_dir):
                return {"status": "error", "message": f"Dataset {dataset_name} not found"}

            ep_files = [f for f in os.listdir(ds_dir) if f.startswith("episode_") and f.endswith(".npz")]
            if len(ep_files) == 0:
                return {"status": "error", "message": "No episodes in dataset to train on"}

            run_id = f"{dataset_name}_act_{time.strftime('%Y%m%d_%H%M%S')}"
            output_dir = os.path.join(MODELS_DIR, run_id)
            os.makedirs(output_dir, exist_ok=True)

            self.state["is_training"] = True
            self.state["dataset"] = dataset_name
            self.state["step"] = 0
            self.state["total_steps"] = steps
            self.state["loss"] = 0.0
            self.state["lr"] = lr
            self.state["fps"] = 0.0
            self.state["eta"] = "--:--:--"
            self.state["logs"].clear()
            self.state["output_dir"] = output_dir
            self.state["error"] = None
            self.state["start_time"] = time.time()

            # Python executable from current lerobot env
            python_bin = sys.executable

            # Wrap with macOS caffeinate to prevent sleep while training
            caffeinate_bin = "/usr/bin/caffeinate"
            base_cmd = [
                python_bin, "-u", os.path.join(BASE_DIR, "train_act_worker.py"),
                "--dataset_dir", ds_dir,
                "--output_dir", output_dir,
                "--steps", str(steps),
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--device", device,
                "--chunk_size", str(chunk_size)
            ]

            if os.path.exists(caffeinate_bin):
                cmd = [caffeinate_bin, "-dimsu"] + base_cmd
            else:
                cmd = base_cmd

            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            except Exception as e:
                self.state["is_training"] = False
                self.state["error"] = str(e)
                return {"status": "error", "message": str(e)}

            threading.Thread(target=self._log_reader_loop, args=(dataset_name, steps, batch_size, lr, device, chunk_size), daemon=True).start()
            print(f"[Train] Launched ACT training on {dataset_name} ({steps} steps) -> {output_dir}")
            return {"status": "started", "output_dir": output_dir, "total_steps": steps}

    def _log_reader_loop(self, dataset_name="", steps=0, batch_size=8, lr=2e-4, device="mps", chunk_size=30):
        proc = self.proc
        for line in proc.stdout:
            line_str = line.strip()
            if not line_str:
                continue

            with self.lock:
                self.state["logs"].append(line_str)
                try:
                    if line_str.startswith("{") and line_str.endswith("}"):
                        data = json.loads(line_str)
                        if data.get("status") == "progress":
                            self.state["step"] = data.get("step", self.state["step"])
                            self.state["loss"] = data.get("loss", self.state["loss"])
                            self.state["lr"] = data.get("lr", self.state["lr"])
                            self.state["fps"] = data.get("fps", self.state["fps"])
                            self.state["eta"] = data.get("eta", self.state["eta"])
                        elif data.get("status") == "completed":
                            self.state["step"] = self.state["total_steps"]
                            self.state["eta"] = "00:00:00"
                except Exception:
                    pass

        proc.wait()
        with self.lock:
            self.state["is_training"] = False
            if proc.returncode != 0 and not self.state.get("error"):
                self.state["error"] = f"Process exited with code {proc.returncode}"
            print(f"[Train] Training process finished with code {proc.returncode}")

        # Auto-chain: if 5000 steps just finished successfully, auto-launch 25000 steps deep training!
        if proc.returncode == 0 and steps == 5000:
            print("[Train] 5000-step test model complete! Auto-launching 25,000-step deep training run in 3 seconds...")
            time.sleep(3)
            self.start_training(dataset_name, steps=25000, batch_size=batch_size, lr=lr, device=device, chunk_size=chunk_size)

    def stop_training(self):
        with self.lock:
            if self.proc and self.state["is_training"]:
                try:
                    self.proc.terminate()
                    time.sleep(0.5)
                    if self.proc.poll() is None:
                        self.proc.kill()
                except Exception as e:
                    print(f"[Train] Error killing process: {e}")
                self.state["is_training"] = False
                self.state["logs"].append("🛑 Training stopped by user.")
                return {"status": "stopped"}
            return {"status": "not_running"}

    def get_status(self):
        with self.lock:
            if self.state["is_training"]:
                return {
                    "is_training": True,
                    "dataset": self.state["dataset"],
                    "step": self.state["step"],
                    "total_steps": self.state["total_steps"],
                    "loss": self.state["loss"],
                    "lr": self.state["lr"],
                    "fps": self.state["fps"],
                    "eta": self.state["eta"],
                    "logs": list(self.state["logs"])[-50:],
                    "output_dir": self.state["output_dir"],
                    "error": self.state["error"]
                }

        # Check if background training is running independently
        try:
            # Check for running train_act_worker
            out = subprocess.check_output(["ps", "aux"]).decode("utf-8")
            if "train_act_worker.py" in out:
                # Find latest training log
                latest_log = None
                latest_mtime = 0
                for root, _, files in os.walk(MODELS_DIR):
                    for f in files:
                        if f.endswith(".log"):
                            p = os.path.join(root, f)
                            mt = os.path.getmtime(p)
                            if mt > latest_mtime:
                                latest_mtime = mt
                                latest_log = p

                if latest_log and (time.time() - latest_mtime < 120):
                    with open(latest_log, "r") as f_log:
                        lines = f_log.readlines()

                    last_progress = None
                    recent_logs = []
                    for line in lines[-50:]:
                        line_str = line.strip()
                        if line_str:
                            recent_logs.append(line_str)
                        if line_str.startswith("{") and "progress" in line_str:
                            try:
                                last_progress = json.loads(line_str)
                            except Exception:
                                pass

                    if last_progress:
                        return {
                            "is_training": True,
                            "dataset": "pick_ball_so101",
                            "step": last_progress.get("step", 0),
                            "total_steps": last_progress.get("total_steps", 50000),
                            "loss": last_progress.get("loss", 0.0),
                            "lr": last_progress.get("lr", 0.0),
                            "fps": last_progress.get("fps", 0.0),
                            "eta": last_progress.get("eta", "--:--:--"),
                            "logs": recent_logs[-50:],
                            "output_dir": os.path.dirname(latest_log),
                            "error": None
                        }
        except Exception:
            pass

        with self.lock:
            return {
                "is_training": False,
                "dataset": self.state["dataset"],
                "step": self.state["step"],
                "total_steps": self.state["total_steps"],
                "loss": self.state["loss"],
                "lr": self.state["lr"],
                "fps": self.state["fps"],
                "eta": self.state["eta"],
                "logs": list(self.state["logs"])[-50:],
                "output_dir": self.state["output_dir"],
                "error": self.state["error"]
            }


# ─────────────────────────────────────────────────────────────────────────────
# AUTONOMOUS INFERENCE / POLICY PLAYER
# ─────────────────────────────────────────────────────────────────────────────
class AutonomousEvaluator:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_evaluating = False
        self.model_path = ""
        self.stop_event = threading.Event()
        self.thread = None
        self.fps = 0.0
        self.current_action = {}
        self.predicted_trajectory = []
        self.trajectory_history = []
        self.error = None

    @staticmethod
    def list_models():
        models = []
        if not os.path.exists(MODELS_DIR):
            return models

        for run_name in sorted(os.listdir(MODELS_DIR), reverse=True):
            run_path = os.path.join(MODELS_DIR, run_name)
            if os.path.isdir(run_path):
                ckpts = [f for f in os.listdir(run_path) if f.endswith(".pt")]
                for ckpt in sorted(ckpts):
                    models.append({
                        "name": f"{run_name} / {ckpt}",
                        "path": os.path.join(run_path, ckpt),
                        "run_name": run_name,
                        "file": ckpt
                    })
        return models

    def start_eval(self, model_path, follower, get_frame_func, joint_names):
        with self.lock:
            if self.is_evaluating:
                return {"status": "error", "message": "Evaluation already running"}

            if not os.path.exists(model_path):
                return {"status": "error", "message": f"Model file not found: {model_path}"}

            self.is_evaluating = True
            self.model_path = model_path
            self.stop_event.clear()
            self.error = None

            self.thread = threading.Thread(
                target=self._eval_loop,
                args=(model_path, follower, get_frame_func, joint_names),
                daemon=True
            )
            self.thread.start()
            print(f"[Eval] Started autonomous policy execution with {model_path}")
            return {"status": "started", "model": model_path}

    def _eval_loop(self, model_path, follower, get_frame_func, joint_names):
        import torch
        import torchvision.transforms as T
        from train_act_worker import SimpleACTPolicy

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            chunk_size = checkpoint.get("args", {}).get("chunk_size", 30)

            # Load dataset z-score normalization stats
            stats = checkpoint.get("stats", {})
            qpos_mean = np.array(stats.get("qpos_mean", [0.0]*len(joint_names)), dtype=np.float32)
            qpos_std = np.array(stats.get("qpos_std", [1.0]*len(joint_names)), dtype=np.float32)
            action_mean = np.array(stats.get("action_mean", [0.0]*len(joint_names)), dtype=np.float32)
            action_std = np.array(stats.get("action_std", [1.0]*len(joint_names)), dtype=np.float32)

            model = SimpleACTPolicy(action_dim=len(joint_names), state_dim=len(joint_names), chunk_size=chunk_size).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[Eval] Model loaded successfully on {device} (chunk_size={chunk_size}, stats={'present' if stats else 'default'}).")
        except Exception as e:
            with self.lock:
                self.is_evaluating = False
                self.error = f"Failed to load model: {e}"
            print(f"[Eval] Load error: {e}")
            return

        # Shared state for Temporal Ensembling between inference and control threads
        chunk_lock = threading.Lock()
        active_chunks = []
        latest_qpos = [0.0] * len(joint_names)

        # Initial read of arm position
        try:
            init_obs = follower.get_observation()
            init_dict = {k: float(v) for k, v in init_obs.items()}
            latest_qpos = [init_dict.get(name, 0.0) for name in joint_names]
        except Exception:
            pass

        # ─── Worker 1: Async Vision & Neural Network Inference Thread ───
        def _inference_worker():
            nonlocal latest_qpos, active_chunks
            print("[Eval] Background Vision & ACT Inference worker active with Temporal Ensembling...")

            while not self.stop_event.is_set():
                try:
                    jpeg_bytes = get_frame_func()
                    if jpeg_bytes is None:
                        time.sleep(0.02)
                        continue

                    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    # Closed-loop feedback: read real physical arm position
                    try:
                        obs = follower.get_observation()
                        if obs:
                            with chunk_lock:
                                latest_qpos = [float(obs.get(name, latest_qpos[i])) for i, name in enumerate(joint_names)]
                    except Exception:
                        pass

                    with chunk_lock:
                        q_curr = np.array(latest_qpos, dtype=np.float32)

                    # Normalize qpos input
                    norm_q = (q_curr - qpos_mean) / qpos_std
                    qpos_tensor = torch.tensor([norm_q], dtype=torch.float32).to(device)

                    with torch.no_grad():
                        pred_norm_chunk = model(img_tensor, qpos_tensor)[0].cpu().numpy()

                    # De-normalize predicted action chunk to physical joint degrees
                    pred_chunk = pred_norm_chunk * action_std + action_mean

                    now = time.time()
                    with chunk_lock:
                        # Append new chunk to temporal ensemble buffer (retain last 1.5 seconds of chunks)
                        active_chunks.append({
                            "chunk": pred_chunk,
                            "start_time": now
                        })
                        # Prune expired chunks older than 1.5s
                        while active_chunks and (now - active_chunks[0]["start_time"] > 1.2):
                            active_chunks.pop(0)

                except Exception as ex:
                    print(f"[Eval-Infer] Error: {ex}")

                # Query policy every ~25ms (~40 FPS)
                time.sleep(0.025)

            print("[Eval] Inference worker exited.")

        infer_thread = threading.Thread(target=_inference_worker, daemon=True)
        infer_thread.start()

        # ─── Worker 2: High-Speed 50Hz Temporal Ensembling Motor Control Loop ───
        CONTROL_HZ = 50.0
        dt = 1.0 / CONTROL_HZ
        smoothed_target = np.array(latest_qpos, dtype=np.float32)
        fps_timer = time.time()
        frames_count = 0
        TRAJ_FPS = 30.0
        ENSEMBLE_EXP_WEIGHT = 0.03

        print(f"[Eval] Temporal Ensembling motor control loop active @ {CONTROL_HZ}Hz...")

        while not self.stop_event.is_set():
            loop_start = time.time()

            with chunk_lock:
                chunks_snapshot = list(active_chunks)

            if chunks_snapshot:
                weights_sum = 0.0
                weighted_action_sum = np.zeros(len(joint_names), dtype=np.float32)

                for item in chunks_snapshot:
                    c = item["chunk"]
                    t_start = item["start_time"]
                    age = max(0.0, loop_start - t_start)
                    step_idx = int(age * TRAJ_FPS)

                    if step_idx < len(c):
                        w = np.exp(-ENSEMBLE_EXP_WEIGHT * step_idx)
                        weighted_action_sum += w * c[step_idx]
                        weights_sum += w

                if weights_sum > 0:
                    raw_target = weighted_action_sum / weights_sum
                else:
                    raw_target = chunks_snapshot[-1]["chunk"][-1]

                # Slight EMA blend (0.6) for micro-jitter filtration
                smoothed_target = smoothed_target + 0.6 * (raw_target - smoothed_target)

                # Send smoothed motor action to follower arm
                action_dict = {joint_names[i]: float(smoothed_target[i]) for i in range(len(joint_names))}
                tensor_action = {k: torch.tensor(v, dtype=torch.float32) for k, v in action_dict.items()}

                try:
                    follower.send_action(tensor_action)
                    with chunk_lock:
                        latest_qpos = smoothed_target.tolist()

                    with self.lock:
                        self.current_action = action_dict
                        if chunks_snapshot:
                            self.predicted_trajectory = chunks_snapshot[-1]["chunk"].round(1).tolist()
                        self.trajectory_history.append(action_dict)
                        if len(self.trajectory_history) > 100:
                            self.trajectory_history.pop(0)

                except Exception as err:
                    print(f"[Eval-Control] Motor write error: {err}")

            frames_count += 1
            if time.time() - fps_timer >= 1.0:
                self.fps = round(frames_count / (time.time() - fps_timer), 1)
                frames_count = 0
                fps_timer = time.time()

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

        infer_thread.join(timeout=0.5)
        with self.lock:
            self.is_evaluating = False
        print("[Eval] Autonomous policy execution ended.")

    def stop_eval(self):
        self.stop_event.set()
        with self.lock:
            self.is_evaluating = False
        return {"status": "stopped"}

    def get_status(self):
        with self.lock:
            return {
                "is_evaluating": self.is_evaluating,
                "model_path": self.model_path,
                "fps": self.fps,
                "current_action": self.current_action,
                "predicted_trajectory": self.predicted_trajectory,
                "trajectory_history": self.trajectory_history[-30:],
                "error": self.error
            }

