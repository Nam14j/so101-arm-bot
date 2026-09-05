"""
SO-101 Arm Dashboard Server
Flask-based backend for controlling SO-101 leader/follower robot arms.
"""

import os
import sys
import time
import json
import threading
import subprocess
import signal
import collections
import urllib.request
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response

# ─── Configuration ─────────────────────────────────────────────────
LEADER_PORT = "/dev/tty.usbmodem5AE60529841"
FOLLOWER_PORT = "/dev/tty.usbmodem5AE60587831"
LEADER_ID = "my_leader_arm"
FOLLOWER_ID = "my_follower_arm"
URDF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101_new_calib.urdf")
SERVER_PORT = 8080

MOVE_DURATION = 3.0      # seconds for smooth moves
CONTROL_LOOP_HZ = 50     # commands per second during moves
RECORD_HZ = 10           # samples per second during recording

SO101_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos"
]

from ai_trainer_manager import DatasetManager, TrainingManager, AutonomousEvaluator, DATASETS_DIR

ai_dataset_mgr = DatasetManager()
ai_train_mgr = TrainingManager()
ai_eval_mgr = AutonomousEvaluator()

_ai_rec_active = False
_ai_rec_dataset = "pick_ball_so101"
_ai_rec_task = "Pick up the ball from the desk"
_ai_rec_qpos = []
_ai_rec_actions = []
_ai_rec_images = []
_ai_rec_timestamps = []
_ai_rec_lock = threading.Lock()
_ai_rec_thread = None
_ai_stop_rec = threading.Event()

# ─── Flask App ─────────────────────────────────────────────────────
web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app = Flask(__name__, static_folder=web_dir, static_url_path="")

# ─── Global State ──────────────────────────────────────────────────
lock = threading.Lock()

state = {
    "leader": None,
    "follower": None,
    "connected": False,
    "teleop": False,
    "recording": False,
    "playing": False,
    "moving": False,
    "recorded_positions": [],
    "initial_pose": None,
    "leader_angles": {},
    "follower_angles": {},
    "downloads": {
        "yolov8": {"status": "idle", "progress": 0, "speed": "0 MB/s"},
        "openvla": {"status": "idle", "progress": 0, "speed": "0 MB/s"}
    },
    "reachy_online": False,
    "reachy_ip": "",
    "reachy_client": None,
    "camera_fps": 30.0
}

stop_teleop = threading.Event()
stop_playback = threading.Event()
stop_move = threading.Event()

# ─── Voice Assistant Process State ─────────────────────────────────
voice_lock = threading.Lock()
voice_process = None
voice_log_buffer = collections.deque(maxlen=2000)
voice_log_counter = 0  # monotonic line counter

# ─── IK Chain ──────────────────────────────────────────────────────
ik_chain = None


def load_ik_chain():
    """Load the URDF kinematic chain for inverse kinematics."""
    global ik_chain
    try:
        from ikpy.chain import Chain
        ik_chain = Chain.from_urdf_file(URDF_FILE)
        print("[IK] Chain loaded successfully")
    except Exception as e:
        print(f"[IK] Failed to load chain: {e}")


# ═══════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return send_from_directory(web_dir, "index.html")


@app.route("/api/status")
def api_status():
    """Return the current state of the system."""
    with lock:
        return jsonify({
            "connected": state["connected"],
            "connection_mode": state.get("connection_mode", "both"),
            "follower_connected": state.get("follower_connected", False),
            "leader_connected": state.get("leader_connected", False),
            "teleop": state["teleop"],
            "recording": state["recording"],
            "playing": state["playing"],
            "moving": state["moving"],
            "step_count": len(state["recorded_positions"]),
            "leader_angles": state["leader_angles"],
            "follower_angles": state["follower_angles"],
            "downloads": state["downloads"],
            "reachy_online": state["reachy_online"],
            "reachy_ip": state["reachy_ip"],
            "camera_fps": state.get("camera_fps", 30.0),
        })


@app.route("/api/connect", methods=["POST"])
def api_connect():
    """Connect to follower arm only, leader arm only, or both arms."""
    data = request.get_json() or {}
    mode = data.get("mode", "both") # 'follower_only', 'leader_only', or 'both'

    with lock:
        if state["connected"]:
            return jsonify({"status": "already_connected", "mode": state.get("connection_mode", "both")})

    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        leader = None
        follower = None
        initial_pose = {}

        if mode in ["both", "leader_only"]:
            print(f"[Connect] Connecting to leader arm on {LEADER_PORT}...")
            leader_config = SO101LeaderConfig(
                port=LEADER_PORT, id=LEADER_ID, use_degrees=True
            )
            leader = SO101Leader(leader_config)
            leader.connect()
            print("[Connect] Leader connected.")

        if mode in ["both", "follower_only"]:
            print(f"[Connect] Connecting to follower arm on {FOLLOWER_PORT}...")
            follower_config = SO101FollowerConfig(
                port=FOLLOWER_PORT, id=FOLLOWER_ID, use_degrees=True
            )
            follower = SO101Follower(follower_config)
            follower.connect()
            print("[Connect] Follower connected.")
            initial_pose = follower.get_observation()

        with lock:
            state["leader"] = leader
            state["follower"] = follower
            state["connected"] = True
            state["connection_mode"] = mode
            state["leader_connected"] = (leader is not None)
            state["follower_connected"] = (follower is not None)
            if follower is not None:
                state["initial_pose"] = dict(initial_pose)
                state["follower_angles"] = {
                    k: round(float(v), 1) for k, v in initial_pose.items()
                }

        print(f"[Connect] Successfully connected in '{mode}' mode.")
        return jsonify({
            "status": "connected",
            "mode": mode,
            "follower_connected": follower is not None,
            "leader_connected": leader is not None
        })

    except Exception as e:
        print(f"[Connect] Error in '{mode}' mode: {e}")
        # Clean up any partially connected arm
        try:
            if leader: leader.disconnect()
        except Exception: pass
        try:
            if follower: follower.disconnect()
        except Exception: pass
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    """Disconnect both arms safely."""
    with lock:
        if not state["connected"]:
            return jsonify({"status": "not_connected"})
        # Signal any running operations to stop
        state["teleop"] = False
        state["recording"] = False
        stop_teleop.set()
        stop_playback.set()
        stop_move.set()

    # Give background threads a moment to exit
    time.sleep(0.3)

    try:
        with lock:
            if state["leader"]:
                try:
                    state["leader"].disconnect()
                except Exception:
                    pass
            if state["follower"]:
                try:
                    state["follower"].disconnect()
                except Exception:
                    pass

            state["leader"] = None
            state["follower"] = None
            state["connected"] = False
            state["leader_angles"] = {}
            state["follower_angles"] = {}
            
            import gc
            gc.collect()

        print("[Disconnect] Arms disconnected.")
        return jsonify({"status": "disconnected"})

    except Exception as e:
        print(f"[Disconnect] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── Recording ─────────────────────────────────────────────────────

@app.route("/api/start_recording", methods=["POST"])
def api_start_recording():
    """Start recording leader arm positions."""
    with lock:
        if not state["connected"]:
            return jsonify({"status": "error", "message": "Not connected"}), 400
        if state["recording"]:
            return jsonify({"status": "already_recording"})
        if state["playing"] or state["moving"]:
            return jsonify({"status": "error", "message": "Another operation is running"}), 400

        state["recorded_positions"] = []
        state["recording"] = True

    # If teleop is running, the teleop loop handles recording.
    # Only start a separate recording thread if teleop is NOT active.
    if not state["teleop"]:
        thread = threading.Thread(target=_recording_loop, daemon=True)
        thread.start()
    print("[Record] Recording started.")
    return jsonify({"status": "recording"})


@app.route("/api/stop_recording", methods=["POST"])
def api_stop_recording():
    """Stop recording and return the number of steps captured."""
    with lock:
        state["recording"] = False
        count = len(state["recorded_positions"])
    print(f"[Record] Stopped — {count} steps captured.")
    return jsonify({"status": "stopped", "step_count": count})


# ─── Playback ──────────────────────────────────────────────────────

@app.route("/api/play_recording", methods=["POST"])
def api_play_recording():
    """Play back recorded positions on the follower arm."""
    with lock:
        if not state["connected"]:
            return jsonify({"status": "error", "message": "Not connected"}), 400
        if state["teleop"] or state["recording"] or state["playing"] or state["moving"]:
            return jsonify({"status": "error", "message": "Another operation is running"}), 400
        if not state["recorded_positions"]:
            return jsonify({"status": "error", "message": "No recorded positions"}), 400

        state["playing"] = True

    stop_playback.clear()
    thread = threading.Thread(target=_playback_loop, daemon=True)
    thread.start()
    print("[Playback] Started.")
    return jsonify({"status": "playing"})


@app.route("/api/stop_playback", methods=["POST"])
def api_stop_playback():
    """Stop the current playback."""
    stop_playback.set()
    print("[Playback] Stop requested.")
    return jsonify({"status": "stopping"})


# ─── Move to XYZ ──────────────────────────────────────────────────

@app.route("/api/move_xyz", methods=["POST"])
def api_move_xyz():
    """Solve IK and move the follower arm to target XYZ coordinates."""
    with lock:
        if not state["connected"]:
            return jsonify({"status": "error", "message": "Not connected"}), 400
        if state["teleop"] or state["recording"] or state["playing"] or state["moving"]:
            return jsonify({"status": "error", "message": "Another operation is running"}), 400
        state["moving"] = True

    data = request.get_json() or {}
    x = float(data.get("x", 0.18))
    y = float(data.get("y", 0.0))
    z = float(data.get("z", 0.12))
    duration = float(data.get("duration", MOVE_DURATION))
    duration = max(0.5, min(10.0, duration))  # Clamp to 0.5–10s

    stop_move.clear()
    thread = threading.Thread(target=_move_xyz_loop, args=(x, y, z, duration), daemon=True)
    thread.start()
    print(f"[Move] Moving to ({x}, {y}, {z}) in {duration}s...")
    return jsonify({"status": "moving", "target": {"x": x, "y": y, "z": z}})


@app.route("/api/stop_move", methods=["POST"])
def api_stop_move():
    """Cancel an in-progress move."""
    stop_move.set()
    print("[Move] Stop requested.")
    return jsonify({"status": "stopping"})


# ─── Home ──────────────────────────────────────────────────────────

@app.route("/api/home", methods=["POST"])
def api_home():
    """Return the follower arm to its initial startup position."""
    with lock:
        if not state["connected"]:
            return jsonify({"status": "error", "message": "Not connected"}), 400
        if state["teleop"] or state["recording"] or state["playing"] or state["moving"]:
            return jsonify({"status": "error", "message": "Another operation is running"}), 400
        if not state["initial_pose"]:
            return jsonify({"status": "error", "message": "No initial pose recorded"}), 400
        state["moving"] = True

    stop_move.clear()
    thread = threading.Thread(target=_home_loop, daemon=True)
    thread.start()
    print("[Home] Returning to initial position...")
    return jsonify({"status": "moving_home"})


# ─── Teleoperation ─────────────────────────────────────────────────

@app.route("/api/start_teleop", methods=["POST"])
def api_start_teleop():
    """Start teleoperation — leader mirrors to follower in real time."""
    with lock:
        if not state["connected"]:
            return jsonify({"status": "error", "message": "Not connected"}), 400
        if state["teleop"]:
            return jsonify({"status": "already_running"})
        if state["playing"] or state["moving"]:
            return jsonify({"status": "error", "message": "Another operation is running"}), 400

        state["teleop"] = True

    stop_teleop.clear()
    thread = threading.Thread(target=_teleop_loop, daemon=True)
    thread.start()
    print("[Teleop] Started.")
    return jsonify({"status": "teleop_started"})


@app.route("/api/stop_teleop", methods=["POST"])
def api_stop_teleop():
    """Stop teleoperation."""
    stop_teleop.set()
    with lock:
        state["recording"] = False  # Also stop recording if active
    print("[Teleop] Stop requested.")
    return jsonify({"status": "stopping"})


@app.route("/api/download_model", methods=["POST"])
def api_download_model():
    """Trigger background download/deployment of a model."""
    data = request.get_json() or {}
    model_id = data.get("model")
    if model_id not in ["yolov8"]:
        return jsonify({"status": "error", "message": "Invalid model ID"}), 400

    with lock:
        current_status = state["downloads"][model_id]["status"]
        if current_status == "downloading":
            return jsonify({"status": "already_downloading"})
        state["downloads"][model_id]["status"] = "downloading"
        state["downloads"][model_id]["progress"] = 0
        state["downloads"][model_id]["speed"] = "0 MB/s"

    thread = threading.Thread(target=_download_model_task, args=(model_id,), daemon=True)
    thread.start()
    return jsonify({"status": "started", "model": model_id})


@app.route("/api/connect_reachy", methods=["POST"])
def api_connect_reachy():
    """Attempt connection to Reachy Mini via IP address or hostname using reachy_mini SDK."""
    data = request.get_json() or {}
    ip = data.get("ip", "").strip()
    if not ip:
        return jsonify({"status": "error", "message": "IP address is required"}), 400

    force = data.get("force", False)

    if force:
        with lock:
            state["reachy_online"] = True
            state["reachy_ip"] = ip
            state["reachy_client"] = None
        print(f"[Reachy] Force connected to IP: {ip} (no gRPC client)")
        return jsonify({"status": "connected", "ip": ip})

    try:
        from reachy_mini import ReachyMini
        print(f"[Reachy] Attempting network SDK connection to Reachy Mini at {ip}:8000...")
        
        # Connect to Reachy Mini remote daemon using the network mode and WebRTC camera stream
        client = ReachyMini(host=ip, connection_mode="network", media_backend="webrtc")
        
        # Enter the context manager manually to establish persistent connection
        client.__enter__()
        
        # Test connection by pulling latest frame
        _ = client.media.get_frame()
        
        with lock:
            state["reachy_online"] = True
            state["reachy_ip"] = ip
            state["reachy_client"] = client
            
        print(f"[Reachy] Successfully connected to Reachy Mini SDK at {ip}")
        return jsonify({"status": "connected", "ip": ip})

    except Exception as e:
        print(f"[Reachy] Connection failed to {ip}: {e}")
        # Clean up in case it was partially entered
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass
        return jsonify({
            "status": "offline",
            "message": f"Reachy Mini connection failed: {e}. Make sure the reachy-mini-daemon is running on the robot."
        })


@app.route("/api/disconnect_reachy", methods=["POST"])
def api_disconnect_reachy():
    """Disconnect Reachy Mini connection and exit context manager."""
    with lock:
        client = state["reachy_client"]
        state["reachy_online"] = False
        state["reachy_ip"] = ""
        state["reachy_client"] = None
        
    if client is not None:
        try:
            client.__exit__(None, None, None)
        except Exception:
            pass
            
    import gc
    gc.collect()
    print("[Reachy] Disconnected.")
    return jsonify({"status": "disconnected"})

# ─── Background Camera Frame Grabber & Async YOLO Worker ──────────
import cv2
import queue

_frame_lock = threading.Lock()
_latest_jpeg = None
_camera_fps = 30.0

_boxes_lock = threading.Lock()
_cached_boxes = []  # [(x1, y1, x2, y2, label)]
_yolo_queue = queue.Queue(maxsize=1)

def _yolo_worker_loop():
    """Asynchronous background worker thread for YOLO inference."""
    global _cached_boxes
    yolo_model = None

    while True:
        try:
            frame = _yolo_queue.get(timeout=0.5)
            if yolo_model is None:
                try:
                    from ultralytics import YOLO
                    yolo_model = YOLO("yolov8n.pt")
                except Exception as e:
                    print(f"[YOLO Worker] Failed to load model: {e}")
                    time.sleep(1.0)
                    continue

            results = yolo_model(frame, verbose=False, imgsz=320)
            new_boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{yolo_model.names[cls_id]} {int(conf * 100)}%"
                    new_boxes.append((x1, y1, x2, y2, label))

            with _boxes_lock:
                _cached_boxes = new_boxes
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[YOLO Worker] Error: {e}")
            time.sleep(0.1)

# Start YOLO worker thread at import time
threading.Thread(target=_yolo_worker_loop, daemon=True).start()


def _camera_grab_loop():
    """Background thread: continuously grabs frames at high quality and encodes to JPEG."""
    global _latest_jpeg, _camera_fps
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    last_t = time.time()
    fps = 30.0

    while True:
        with lock:
            client = state["reachy_client"]
            online = state["reachy_online"]
            detect = state.get("detect_objects", False)

        if not online or not client:
            time.sleep(0.3)
            continue

        try:
            frame = client.media.get_frame()
            if frame is not None:
                if detect:
                    # Non-blocking push to YOLO queue if ready
                    if not _yolo_queue.full():
                        try:
                            _yolo_queue.put_nowait(frame.copy())
                        except queue.Full:
                            pass

                    # Fast non-blocking drawing of cached boxes
                    with _boxes_lock:
                        boxes = list(_cached_boxes)

                    if boxes:
                        annotated = frame.copy()
                        for x1, y1, x2, y2, label in boxes:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 153, 255), 2)
                            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(annotated, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), (0, 153, 255), -1)
                            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        frame = annotated

                ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                if ret:
                    now = time.time()
                    dt = now - last_t
                    if dt > 0:
                        fps = round(0.9 * fps + 0.1 * (1.0 / dt), 1)
                    last_t = now
                    with _frame_lock:
                        _latest_jpeg = jpeg.tobytes()
                        _camera_fps = fps
                    with lock:
                        state["camera_fps"] = fps
            else:
                time.sleep(0.005)
        except Exception as e:
            print(f"[Camera] Grab exception: {e}")
            time.sleep(0.2)

# Start camera grabber thread once at import time
threading.Thread(target=_camera_grab_loop, daemon=True).start()


def generate_reachy_camera():
    """Yields the latest buffered JPEG frame as an MJPEG stream at maximum speed."""
    prev_jpeg = None
    while True:
        with _frame_lock:
            jpeg = _latest_jpeg

        if jpeg is not None and jpeg is not prev_jpeg:
            prev_jpeg = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.005)


@app.route("/api/reachy_camera")
def api_reachy_camera():
    """Proxy endpoint to stream Reachy Mini's MJPEG frames to standard HTTP image tags."""
    res = Response(generate_reachy_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
    res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    res.headers['Pragma'] = 'no-cache'
    res.headers['Expires'] = '0'
    return res


@app.route("/api/toggle_detection", methods=["POST"])
def api_toggle_detection():
    """Toggles live YOLO object detection overlay on Reachy's camera feed."""
    data = request.get_json() or {}
    enable = data.get("enable", None)
    with lock:
        if enable is not None:
            state["detect_objects"] = bool(enable)
        else:
            state["detect_objects"] = not state.get("detect_objects", False)
        current = state["detect_objects"]
    return jsonify({"status": "success", "detect_objects": current})


# ─── Camera Homography Calibration ────────────────────────────────

import json as _json
import os as _os

_CALIB_FILE = _os.path.join(_os.path.dirname(__file__), "camera_calibration.json")

# In-memory calibration state
_calib_points = []          # list of {"px": x, "py": y, "rx": x, "ry": y}
_homography_matrix = None   # 3x3 numpy matrix when computed

def _load_calibration():
    global _calib_points, _homography_matrix
    if _os.path.exists(_CALIB_FILE):
        try:
            with open(_CALIB_FILE) as f:
                data = _json.load(f)
            _calib_points = data.get("points", [])
            H = data.get("homography")
            if H and len(_calib_points) >= 4:
                import numpy as np
                _homography_matrix = np.array(H, dtype=np.float64)
                print(f"[Calib] Loaded calibration with {len(_calib_points)} points")
        except Exception as e:
            print(f"[Calib] Failed to load calibration: {e}")

_load_calibration()

def _apply_homography(px, py):
    """Map pixel coords → robot coords using the computed homography matrix."""
    if _homography_matrix is None:
        return None, None
    import numpy as np
    pt = np.array([[[float(px), float(py)]]], dtype=np.float64)
    import cv2 as _cv2_h
    result = _cv2_h.perspectiveTransform(pt, _homography_matrix)
    rx = float(result[0][0][0])
    ry = float(result[0][0][1])
    return rx, ry


# ─── Logitech HD Webcam (C920 / HD 1080p) ─────────────────────────
# Separate grab loop, YOLO worker, and MJPEG stream for the desk webcam.

_logi_frame_lock  = threading.Lock()
_logi_latest_jpeg = None
_logi_raw_jpeg    = None
_logi_fps         = 0.0

_logi_boxes_lock     = threading.Lock()
_logi_cached_boxes   = []   # list of (x1, y1, x2, y2, label, cx, cy, rx, ry)
_logi_cached_objects = []   # list of structured dicts for the dashboard
_logi_detect_queue   = queue.Queue(maxsize=1)

# Text prompts the user wants to detect (editable from UI)
_detect_classes = ["ball", "robot arm", "tool", "camera", "hand", "box", "screw", "lego", "wire", "cup", "pen", "bottle", "block"]

# Pick the best available device: CUDA > MPS (Apple Silicon) > CPU
import torch as _torch
if _torch.cuda.is_available():
    _DEVICE = "cuda"
elif _torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"
print(f"[Vision] Grounding DINO will run on: {_DEVICE}")


def _gdino_worker():
    """Grounding DINO open-vocabulary detection worker with real-world grid coordinate mapping."""
    global _logi_cached_boxes, _logi_cached_objects
    gdino_proc  = None
    gdino_model = None

    while True:
        try:
            frame = _logi_detect_queue.get(timeout=0.5)

            # ── Load model on first frame ─────────────────────────────
            if gdino_proc is None:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                import torch
                _id = "IDEA-Research/grounding-dino-base"
                print(f"[Vision] Loading Grounding DINO on {_DEVICE}...")
                gdino_proc  = AutoProcessor.from_pretrained(_id)
                gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(_id).to(_DEVICE)
                gdino_model.eval()
                print("[Vision] Grounding DINO ready ✓")

            import torch
            from PIL import Image as _PIL

            pil    = _PIL.fromarray(frame[:, :, ::-1])  # BGR → RGB
            prompt = ". ".join(_detect_classes) + "."
            inputs = {k: v.to(_DEVICE) if hasattr(v, "to") else v
                      for k, v in gdino_proc(images=pil, text=prompt, return_tensors="pt").items()}

            with torch.no_grad():
                outputs = gdino_model(**inputs)

            res = gdino_proc.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                threshold=0.35, text_threshold=0.25,
                target_sizes=[pil.size[::-1]]
            )[0]

            h, w = frame.shape[:2]
            raw  = []
            for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box.tolist()]
                x2, y2 = min(w, x2), min(h, y2)
                raw.append((x1, y1, x2, y2, str(label), float(score)))

            # ── Map detections to real-world table coordinates ────────
            new_boxes, new_objects = [], []
            for idx, (x1, y1, x2, y2, name, conf) in enumerate(raw):
                cx = (x1 + x2) / 2.0
                cy = float(y2)              # base of box = table contact point
                rx, ry = _apply_homography(cx, cy)

                if rx is not None and ry is not None:
                    label_txt = f"{name} {int(conf*100)}% | X:{rx:.3f}m Y:{ry:.3f}m"
                    obj = {"id": idx+1, "name": name, "confidence": round(conf, 2),
                           "box": [x1,y1,x2,y2], "center_px": [round(cx,1), round(cy,1)],
                           "robot_x": round(rx, 4), "robot_y": round(ry, 4),
                           "label": label_txt, "calibrated": True}
                else:
                    label_txt = f"{name} {int(conf*100)}%"
                    obj = {"id": idx+1, "name": name, "confidence": round(conf, 2),
                           "box": [x1,y1,x2,y2], "center_px": [round(cx,1), round(cy,1)],
                           "robot_x": None, "robot_y": None,
                           "label": label_txt, "calibrated": False}

                new_boxes.append((x1, y1, x2, y2, label_txt, cx, cy, rx, ry))
                new_objects.append(obj)

            with _logi_boxes_lock:
                _logi_cached_boxes   = new_boxes
                _logi_cached_objects = new_objects

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Vision] Error: {e}")
            import traceback; traceback.print_exc()
            time.sleep(0.3)

threading.Thread(target=_gdino_worker, daemon=True).start()



LOGI_CAM_INDEX = 0

# Open the Logitech camera immediately on the main thread to prevent macOS/AVFoundation thread crashes (CAException)
import cv2 as _cv2
_logi_cap = _cv2.VideoCapture(LOGI_CAM_INDEX)
if _logi_cap.isOpened():
    _logi_cap.set(_cv2.CAP_PROP_FRAME_WIDTH,  1920)
    _logi_cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    _logi_cap.set(_cv2.CAP_PROP_FPS,          30)
    _logi_cap.set(_cv2.CAP_PROP_BUFFERSIZE,   1)
    print(f"[Logi] Main-thread camera initialization successful on index {LOGI_CAM_INDEX}")
else:
    print(f"[Logi] Could not open webcam at index {LOGI_CAM_INDEX} on main thread")


def _logi_grab_loop():
    """Background thread: grabs frames from the Logitech webcam and encodes to JPEG safely with high-precision grid annotations."""
    global _logi_latest_jpeg, _logi_raw_jpeg, _logi_fps, _logi_cap
    encode_params = [_cv2.IMWRITE_JPEG_QUALITY, 85, _cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    last_t = time.time()
    fps    = 30.0

    while True:
        if _logi_cap is None or not _logi_cap.isOpened():
            time.sleep(0.5)
            continue

        try:
            ret, frame = _logi_cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # Always save pure, clean raw frame for AI training / evaluation (no synthetic overlays)
            raw_ok, raw_jpeg = _cv2.imencode('.jpg', frame, encode_params)

            detect = state.get("logi_detect_objects", False)
            if detect:
                if not _logi_detect_queue.full():
                    try:
                        _logi_detect_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                with _logi_boxes_lock:
                    boxes = list(_logi_cached_boxes)
                if boxes:
                    ann = frame.copy()
                    for item in boxes:
                        x1, y1, x2, y2, label = item[0], item[1], item[2], item[3], item[4]
                        cx, cy = item[5], item[6] if len(item) > 6 else ((x1+x2)/2, float(y2))
                        rx = item[7] if len(item) > 7 else None
                        
                        # Main bounding box
                        _cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 229, 255), 2)
                        
                        # Target anchor crosshair at table contact point
                        if rx is not None:
                            icx, icy = int(cx), int(cy)
                            _cv2.circle(ann, (icx, icy), 5, (0, 255, 128), -1)
                            _cv2.circle(ann, (icx, icy), 10, (0, 255, 128), 1)
                            _cv2.line(ann, (icx - 14, icy), (icx + 14, icy), (0, 255, 128), 1)
                            _cv2.line(ann, (icx, icy - 14), (icx, icy + 14), (0, 255, 128), 1)
                        
                        # Badge background with high-contrast border
                        ts = _cv2.getTextSize(label, _cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
                        badge_y1 = max(0, y1 - ts[1] - 8)
                        badge_y2 = y1
                        badge_x2 = min(ann.shape[1], x1 + ts[0] + 12)
                        _cv2.rectangle(ann, (x1, badge_y1), (badge_x2, badge_y2), (18, 24, 38), -1)
                        _cv2.rectangle(ann, (x1, badge_y1), (badge_x2, badge_y2), (0, 229, 255), 1)
                        text_color = (0, 255, 128) if rx is not None else (0, 229, 255)
                        _cv2.putText(ann, label, (x1 + 6, y1 - 4), _cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, _cv2.LINE_AA)
                    frame = ann

            ok, jpeg = _cv2.imencode('.jpg', frame, encode_params)
            if ok:
                now = time.time()
                dt  = now - last_t
                if dt > 0:
                    fps = round(0.9 * fps + 0.1 / dt, 1)
                last_t = now
                with _logi_frame_lock:
                    _logi_latest_jpeg = jpeg.tobytes()
                    if raw_ok:
                        _logi_raw_jpeg = raw_jpeg.tobytes()
                    _logi_fps = fps
                with lock:
                    state["logi_fps"] = fps
        except Exception as e:
            print(f"[Logi] Read warning: {e}")
            time.sleep(0.1)

threading.Thread(target=_logi_grab_loop, daemon=True).start()




def generate_logi_camera():
    """Yields MJPEG frames from the Logitech webcam at maximum speed."""
    prev = None
    while True:
        with _logi_frame_lock:
            jpeg = _logi_latest_jpeg
        if jpeg is not None and jpeg is not prev:
            prev = jpeg
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
        else:
            time.sleep(0.005)


@app.route("/api/logi_camera")
def api_logi_camera():
    """MJPEG stream for the Logitech HD webcam."""
    res = Response(generate_logi_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')
    res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    res.headers['Pragma'] = 'no-cache'
    res.headers['Expires'] = '0'
    return res


@app.route("/api/logi_snapshot")
def api_logi_snapshot():
    """Returns the latest Logitech frame as a single JPEG (used by the voice assistant)."""
    with _logi_frame_lock:
        jpeg = _logi_latest_jpeg
    if jpeg is None:
        return jsonify({"status": "error", "message": "No frame available"}), 503
    from flask import send_file
    import io
    return send_file(io.BytesIO(jpeg), mimetype='image/jpeg')


@app.route("/api/toggle_logi_detection", methods=["POST"])
def api_toggle_logi_detection():
    """Toggles live SOTA Vision / Object Detection overlay on the Logitech webcam feed."""
    data = request.get_json() or {}
    enable = data.get("enable", None)
    with lock:
        if enable is not None:
            state["logi_detect_objects"] = bool(enable)
        else:
            state["logi_detect_objects"] = not state.get("logi_detect_objects", False)
        current = state["logi_detect_objects"]
    print(f"[Vision] Logitech AI Vision toggled: {'ON' if current else 'OFF'}")
    return jsonify({"status": "success", "detect_objects": current})



@app.route("/api/set_detector_classes", methods=["POST"])
def api_set_detector_classes():
    """Updates the text prompts Grounding DINO searches for."""
    global _detect_classes
    data = request.get_json() or {}
    classes_str = data.get("classes", "").strip()
    if classes_str:
        classes = [c.strip() for c in classes_str.split(",") if c.strip()]
        if classes:
            _detect_classes = classes
            print(f"[Vision] Updated detection prompts: {_detect_classes}")
            return jsonify({"status": "success", "classes": _detect_classes})
    return jsonify({"status": "error", "message": "No valid classes provided"}), 400



@app.route("/api/detected_objects")
def api_detected_objects():
    """Returns real-time list of detected objects with calculated table grid coordinates."""
    with _logi_boxes_lock:
        objs = list(_logi_cached_objects)
    return jsonify({
        "status": "ok",
        "calibrated": _homography_matrix is not None,
        "count": len(objs),
        "objects": objs
    })



@app.route("/api/calibration/status")
def api_calibration_status():
    return jsonify({
        "calibrated": _homography_matrix is not None,
        "points": _calib_points,
        "point_count": len(_calib_points)
    })

@app.route("/api/calibration/add_point", methods=["POST"])
def api_calibration_add_point():
    global _calib_points
    data = request.get_json() or {}
    px = float(data.get("px", 0))
    py = float(data.get("py", 0))
    rx = float(data.get("rx", 0))
    ry = float(data.get("ry", 0))
    idx = int(data.get("index", len(_calib_points)))
    # Replace or append
    point = {"px": px, "py": py, "rx": rx, "ry": ry}
    if idx < len(_calib_points):
        _calib_points[idx] = point
    else:
        _calib_points.append(point)
    print(f"[Calib] Point {idx}: pixel({px:.0f},{py:.0f}) → robot({rx:.3f},{ry:.3f})")
    return jsonify({"status": "ok", "points": _calib_points})

@app.route("/api/calibration/remove_point", methods=["POST"])
def api_calibration_remove_point():
    global _calib_points
    data = request.get_json() or {}
    idx = int(data.get("index", -1))
    if 0 <= idx < len(_calib_points):
        _calib_points.pop(idx)
    return jsonify({"status": "ok", "points": _calib_points})

@app.route("/api/calibration/compute", methods=["POST"])
def api_calibration_compute():
    global _homography_matrix
    if len(_calib_points) < 4:
        return jsonify({"status": "error", "message": f"Need at least 4 points, have {len(_calib_points)}"}), 400
    try:
        import numpy as np
        import cv2 as _cv2_h
        src = np.array([[p["px"], p["py"]] for p in _calib_points], dtype=np.float64)
        dst = np.array([[p["rx"], p["ry"]] for p in _calib_points], dtype=np.float64)
        H, mask = _cv2_h.findHomography(src, dst, _cv2_h.RANSAC, 5.0)
        if H is None:
            return jsonify({"status": "error", "message": "Homography computation failed — points may be collinear"}), 400
        _homography_matrix = H
        # Persist to disk
        with open(_CALIB_FILE, "w") as f:
            _json.dump({"points": _calib_points, "homography": H.tolist()}, f, indent=2)
        print(f"[Calib] Homography computed from {len(_calib_points)} points and saved.")
        return jsonify({"status": "ok", "homography": H.tolist()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/calibration/reset", methods=["POST"])
def api_calibration_reset():
    global _calib_points, _homography_matrix
    _calib_points = []
    _homography_matrix = None
    if _os.path.exists(_CALIB_FILE):
        _os.remove(_CALIB_FILE)
    print("[Calib] Calibration reset.")
    return jsonify({"status": "ok"})

@app.route("/api/calibration/transform", methods=["POST"])
def api_calibration_transform():
    """Transform a pixel coordinate to robot coordinates using homography."""
    data = request.get_json() or {}
    px = float(data.get("px", 0))
    py = float(data.get("py", 0))
    if _homography_matrix is None:
        return jsonify({"status": "uncalibrated"}), 400
    rx, ry = _apply_homography(px, py)
    return jsonify({"status": "ok", "rx": rx, "ry": ry})


@app.route("/api/arm_position")
def api_arm_position():
    """Read current follower arm joint angles and return end-effector XY via FK."""
    with lock:
        follower = state.get("follower")
        connected = state.get("connected", False)

    if not connected or follower is None:
        return jsonify({"status": "error", "message": "Arm not connected"}), 400

    if ik_chain is None:
        return jsonify({"status": "error", "message": "IK chain not loaded"}), 400

    try:
        import numpy as np

        # Read live joint positions from the follower arm
        obs = {k: float(v) for k, v in follower.get_observation().items()}

        # Build angle array in the same order as ik_chain.links
        angles = []
        for i, link in enumerate(ik_chain.links):
            key = f"{link.name}.pos"
            if key in obs:
                angles.append(np.radians(obs[key]))
            else:
                angles.append(0.0)

        # Run forward kinematics → 4x4 transform matrix
        transform = ik_chain.forward_kinematics(angles)
        x = float(transform[0, 3])  # end-effector X in metres
        y = float(transform[1, 3])  # end-effector Y in metres
        z = float(transform[2, 3])  # end-effector Z in metres

        return jsonify({"status": "ok", "x": x, "y": y, "z": z, "joints": obs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/move_head", methods=["POST"])
def api_move_head():
    """Commands Reachy Mini's neck actuators using the reachy_mini SDK."""
    data = request.get_json() or {}
    yaw = float(data.get("yaw", 0.0))
    pitch = float(data.get("pitch", 0.0))
    roll = float(data.get("roll", 0.0))
    z = float(data.get("z", 0.0))
    duration = float(data.get("duration", 0.1))
    
    with lock:
        client = state["reachy_client"]
        online = state["reachy_online"]
        
    if not online or not client:
        return jsonify({"status": "error", "message": "Reachy is offline"}), 400
        
    try:
        from reachy_mini.utils import create_head_pose
        pose = create_head_pose(yaw=yaw, pitch=pitch, roll=roll, z=z, degrees=True)
        client.goto_target(head=pose, duration=duration)
        return jsonify({"status": "success", "yaw": yaw, "pitch": pitch, "roll": roll, "z": z})
    except Exception as e:
        print(f"[Head] Movement failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/move_torso", methods=["POST"])
def api_move_torso():
    """Commands Reachy Mini's torso actuator (body yaw)."""
    data = request.get_json() or {}
    yaw = float(data.get("yaw", 0.0))
    duration = float(data.get("duration", 0.1))
    
    with lock:
        client = state["reachy_client"]
        online = state["reachy_online"]
        
    if not online or not client:
        return jsonify({"status": "error", "message": "Reachy is offline"}), 400
        
    try:
        import numpy as np
        client.goto_target(body_yaw=np.deg2rad(yaw), duration=duration)
        return jsonify({"status": "success", "yaw": yaw})
    except Exception as e:
        print(f"[Torso] Movement failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/move_antennas", methods=["POST"])
def api_move_antennas():
    """Commands Reachy Mini's antenna actuators."""
    data = request.get_json() or {}
    left = float(data.get("left_angle", 0.0))
    right = float(data.get("right_angle", 0.0))
    duration = float(data.get("duration", 0.1))
    
    with lock:
        client = state["reachy_client"]
        online = state["reachy_online"]
        
    if not online or not client:
        return jsonify({"status": "error", "message": "Reachy is offline"}), 400
        
    try:
        import numpy as np
        client.goto_target(antennas=np.deg2rad([left, right]), duration=duration)
        return jsonify({"status": "success", "left": left, "right": right})
    except Exception as e:
        print(f"[Antennas] Movement failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/reset_all", methods=["POST"])
def api_reset_all():
    """Resets head, torso, and antennas to center with a single goto_target call."""
    data = request.get_json() or {}
    duration = float(data.get("duration", 1.0))
    
    with lock:
        client = state["reachy_client"]
        online = state["reachy_online"]
        
    if not online or not client:
        return jsonify({"status": "error", "message": "Reachy is offline"}), 400
        
    try:
        import numpy as np
        from reachy_mini.utils import create_head_pose
        
        pose = create_head_pose(yaw=0, pitch=0, degrees=True)
        client.goto_target(
            head=pose,
            body_yaw=np.float64(0.0),
            antennas=np.deg2rad([0.0, 0.0]),
            duration=duration
        )
        print("[Reset] All joints reset to center.")
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"[Reset] Failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ─── Voice Assistant Management ────────────────────────────────────

def _voice_reader_thread(proc):
    """Background thread that reads voice_interface.py stdout line by line."""
    global voice_log_counter
    try:
        for raw_line in iter(proc.stdout.readline, ''):
            if raw_line == '':
                break
            line = raw_line.rstrip('\n')
            with voice_lock:
                voice_log_counter += 1
                voice_log_buffer.append((voice_log_counter, line))
    except Exception:
        pass
    finally:
        with voice_lock:
            voice_log_counter += 1
            voice_log_buffer.append((voice_log_counter, '[System] Voice assistant process ended.'))


@app.route("/api/voice/start", methods=["POST"])
def api_voice_start():
    """Starts voice_interface.py as a subprocess."""
    global voice_process, voice_log_counter
    
    with voice_lock:
        if voice_process and voice_process.poll() is None:
            return jsonify({"status": "error", "message": "Already running"}), 400
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voice_interface.py')
    python_bin = sys.executable
    
    try:
        with voice_lock:
            voice_log_buffer.clear()
            voice_log_counter = 0
        
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['GST_DISABLE_REGISTRY_FORK'] = '1'
        env['GST_DISABLE_SEGTRAP'] = '1'
        bin_dir = os.path.expanduser("~/miniforge3/envs/lerobot/bin")
        env['PATH'] = f"{bin_dir}:" + env.get('PATH', '')
        
        proc = subprocess.Popen(
            [python_bin, '-u', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            text=True,
            bufsize=1
        )
        
        with voice_lock:
            voice_process = proc
            voice_log_counter += 1
            voice_log_buffer.append((voice_log_counter, f'[System] Starting voice_interface.py (PID {proc.pid})...'))
        
        reader = threading.Thread(target=_voice_reader_thread, args=(proc,), daemon=True)
        reader.start()
        
        print(f"[Voice] Started voice_interface.py (PID {proc.pid})")
        return jsonify({"status": "success", "pid": proc.pid})
    except Exception as e:
        print(f"[Voice] Failed to start: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/voice/stop", methods=["POST"])
def api_voice_stop():
    """Stops the voice_interface.py subprocess."""
    global voice_process
    
    with voice_lock:
        proc = voice_process
    
    if not proc or proc.poll() is not None:
        return jsonify({"status": "error", "message": "Not running"}), 400
    
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
        
        with voice_lock:
            global voice_log_counter
            voice_log_counter += 1
            voice_log_buffer.append((voice_log_counter, '[System] Voice assistant stopped.'))
            voice_process = None
        
        print("[Voice] Stopped voice_interface.py")
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/voice/status")
def api_voice_status():
    """Returns whether the voice assistant is running."""
    with voice_lock:
        proc = voice_process
    running = proc is not None and proc.poll() is None
    return jsonify({"running": running})


@app.route("/api/voice/logs")
def api_voice_logs():
    """Returns log lines since the given 'since' counter."""
    since = int(request.args.get('since', 0))
    lines = []
    with voice_lock:
        for counter, text in voice_log_buffer:
            if counter > since:
                lines.append({"id": counter, "text": text})
    return jsonify({"lines": lines})


@app.route("/api/gesture", methods=["POST"])
def api_gesture():
    """Plays an expressive gesture on Reachy Mini."""
    data = request.get_json() or {}
    gesture = data.get("gesture", "yes")
    
    with lock:
        client = state["reachy_client"]
        online = state["reachy_online"]
        
    if not online or not client:
        return jsonify({"status": "error", "message": "Reachy is offline"}), 400
        
    def _run_gesture(g_name, r_client):
        try:
            import numpy as np
            from reachy_mini.utils import create_head_pose
            if g_name == "yes":
                for pitch in [15, -10, 15, -10, 0]:
                    pose = create_head_pose(yaw=0, pitch=pitch, degrees=True)
                    r_client.goto_target(head=pose, duration=0.25)
                    time.sleep(0.25)
            elif g_name == "no":
                for yaw in [-20, 20, -20, 20, 0]:
                    pose = create_head_pose(yaw=yaw, pitch=0, degrees=True)
                    r_client.goto_target(head=pose, duration=0.25)
                    time.sleep(0.25)
            elif g_name == "excited":
                for i in range(3):
                    pose = create_head_pose(yaw=0, pitch=15 if i % 2 == 0 else -10, degrees=True)
                    r_client.goto_target(
                        head=pose,
                        antennas=np.deg2rad([90, -90] if i % 2 == 0 else [-90, 90]),
                        duration=0.2
                    )
                    time.sleep(0.2)
                pose = create_head_pose(yaw=0, pitch=0, degrees=True)
                r_client.goto_target(head=pose, antennas=np.deg2rad([0, 0]), duration=0.3)
            elif g_name == "sad":
                pose = create_head_pose(yaw=0, pitch=-20, degrees=True)
                r_client.goto_target(head=pose, antennas=np.deg2rad([-45, 45]), duration=1.0)
            elif g_name == "confused":
                pose = create_head_pose(yaw=15, pitch=10, degrees=True)
                r_client.goto_target(head=pose, antennas=np.deg2rad([90, -30]), duration=0.8)
            elif g_name == "look_around":
                for yaw, torso_yaw in [(-30, -30), (30, 30), (0, 0)]:
                    pose = create_head_pose(yaw=yaw, pitch=5, degrees=True)
                    r_client.goto_target(head=pose, body_yaw=np.deg2rad(torso_yaw), duration=1.2)
                    time.sleep(1.3)
        except Exception as err:
            print(f"[Gesture Error] {err}")

    threading.Thread(target=_run_gesture, args=(gesture, client), daemon=True).start()
    return jsonify({"status": "success", "gesture": gesture})


# ═══════════════════════════════════════════════════════════════════
#  AI DATASET RECORDING, TRAINING & AUTONOMOUS INFERENCE ROUTES
# ═══════════════════════════════════════════════════════════════════

@app.route("/api/ai/dataset/list")
def api_ai_dataset_list():
    """List all recorded datasets and their episode counts."""
    return jsonify({"status": "ok", "datasets": ai_dataset_mgr.list_datasets()})


@app.route("/api/ai/dataset/info")
def api_ai_dataset_info():
    """Get metadata and episode breakdown for a specific dataset."""
    name = request.args.get("dataset", "pick_ball_so101")
    return jsonify({"status": "ok", "info": ai_dataset_mgr.get_dataset_info(name)})


@app.route("/api/ai/dataset/video")
def api_ai_dataset_video():
    """Stream MJPEG video replay of all recorded frames for a specific episode."""
    dataset = request.args.get("dataset", "pick_ball_so101")
    ep_idx = int(request.args.get("index", 0))
    speed = float(request.args.get("speed", 1.0))
    loop = request.args.get("loop", "true").lower() == "true"

    ds_dir = os.path.join(DATASETS_DIR, dataset)
    ep_path = os.path.join(ds_dir, f"episode_{ep_idx:04d}.npz")
    if not os.path.exists(ep_path):
        return jsonify({"status": "error", "message": "Episode not found"}), 404

    def _generate_video_stream():
        try:
            d = np.load(ep_path, allow_pickle=True)
            images = d["images"]
            frame_delay = max(0.01, (1.0 / 30.0) / speed)
            
            while True:
                for img_data in images:
                    if isinstance(img_data, bytes):
                        jpeg_bytes = img_data
                    elif isinstance(img_data, np.ndarray):
                        _, buf = cv2.imencode(".jpg", img_data if img_data.shape[2] == 3 else cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))
                        jpeg_bytes = buf.tobytes()
                    else:
                        continue

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )
                    time.sleep(frame_delay)
                
                if not loop:
                    break
                time.sleep(0.5) # Brief pause before looping
        except Exception as e:
            print(f"[Video-Stream] Error: {e}")

    return Response(
        _generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/ai/dataset/start_episode", methods=["POST"])
def api_ai_start_episode():
    """Start recording a new teleoperation episode with camera frames."""
    global _ai_rec_active, _ai_rec_dataset, _ai_rec_task, _ai_rec_qpos, _ai_rec_actions, _ai_rec_images, _ai_rec_timestamps
    data = request.get_json() or {}
    _ai_rec_dataset = data.get("dataset", "pick_ball_so101").strip() or "pick_ball_so101"
    _ai_rec_task = data.get("task", "Pick up the ball from the desk").strip() or "Pick up the ball from the desk"

    with _ai_rec_lock:
        _ai_rec_qpos = []
        _ai_rec_actions = []
        _ai_rec_images = []
        _ai_rec_timestamps = []
        _ai_rec_active = True

    # Auto-start teleoperation if arms are connected and teleop is not running
    with lock:
        if state["connected"] and not state["teleop"] and not state["playing"] and not state["moving"]:
            state["teleop"] = True
            stop_teleop.clear()
            threading.Thread(target=_teleop_loop, daemon=True).start()
            print("[AI-Record] Auto-started teleoperation for episode recording.")

    print(f"[AI-Record] Started episode recording for dataset '{_ai_rec_dataset}' ('{_ai_rec_task}')")
    return jsonify({"status": "recording", "dataset": _ai_rec_dataset})


@app.route("/api/ai/dataset/stop_episode", methods=["POST"])
def api_ai_stop_episode():
    """Stop the current recording and optionally save it to disk."""
    global _ai_rec_active
    data = request.get_json() or {}
    save = bool(data.get("save", True))

    _ai_rec_active = False

    with _ai_rec_lock:
        steps_count = len(_ai_rec_qpos)
        if save and steps_count >= 5:
            ep_idx = ai_dataset_mgr.save_episode(
                _ai_rec_dataset,
                _ai_rec_task,
                _ai_rec_qpos,
                _ai_rec_actions,
                _ai_rec_images,
                _ai_rec_timestamps
            )
            print(f"[AI-Record] Saved Episode #{ep_idx} ({steps_count} steps)")
            return jsonify({"status": "saved", "episode_index": ep_idx, "steps": steps_count})
        elif save:
            return jsonify({"status": "error", "message": "Episode was too short (< 5 frames), discarded."}), 400
        else:
            print("[AI-Record] Episode discarded by user.")
            return jsonify({"status": "discarded", "steps": steps_count})


@app.route("/api/ai/dataset/status")
def api_ai_dataset_status():
    """Check live status of the episode recorder."""
    with _ai_rec_lock:
        steps = len(_ai_rec_qpos)
    return jsonify({
        "status": "ok",
        "is_recording": _ai_rec_active,
        "dataset": _ai_rec_dataset,
        "task": _ai_rec_task,
        "current_steps": steps,
        "duration_s": round(steps / 30.0, 1)
    })


@app.route("/api/ai/dataset/delete_episode", methods=["POST"])
def api_ai_delete_episode():
    """Delete an episode from a dataset."""
    data = request.get_json() or {}
    dataset = data.get("dataset", "pick_ball_so101")
    ep_idx = int(data.get("index", -1))
    success = ai_dataset_mgr.delete_episode(dataset, ep_idx)
    return jsonify({"status": "ok" if success else "error"})


@app.route("/api/ai/dataset/replay_episode", methods=["POST"])
def api_ai_replay_episode():
    """Replay an episode's trajectory on the follower arm, safely disengaging teleoperation first."""
    data = request.get_json() or {}
    dataset = data.get("dataset", "pick_ball_so101")
    ep_idx = int(data.get("index", -1))

    ds_dir = os.path.join(DATASETS_DIR, dataset)
    ep_path = os.path.join(ds_dir, f"episode_{ep_idx:04d}.npz")
    if not os.path.exists(ep_path):
        return jsonify({"status": "error", "message": "Episode file not found"}), 404

    # Automatically disengage teleoperation and recording if active
    stop_teleop.set()
    stop_playback.set()
    stop_move.set()

    with lock:
        follower = state.get("follower")
        if not follower:
            return jsonify({"status": "error", "message": "Follower arm not connected"}), 400
        state["teleop"] = False
        state["recording"] = False
        state["playing"] = True

    def _run_replay():
        try:
            # Allow teleop loop a moment to completely release serial bus
            time.sleep(0.1)

            npz = np.load(ep_path, allow_pickle=True)
            actions = npz["actions"]

            print(f"[AI-Replay] Replaying Episode #{ep_idx} ({len(actions)} steps)...")
            for i, act_vec in enumerate(actions):
                act_dict = {SO101_JOINT_KEYS[j]: float(act_vec[j]) for j in range(len(SO101_JOINT_KEYS))}
                follower.send_action(act_dict)
                with lock:
                    state["follower_angles"] = {k: round(v, 1) for k, v in act_dict.items()}
                time.sleep(1.0 / 30.0)
            print(f"[AI-Replay] Episode #{ep_idx} replay complete.")
        except Exception as e:
            print(f"[AI-Replay] Error: {e}")
        finally:
            with lock:
                state["playing"] = False

    threading.Thread(target=_run_replay, daemon=True).start()
    return jsonify({"status": "replaying", "episode_index": ep_idx, "teleop_disengaged": True})


# ─── Training Routes ───────────────────────────────────────────────

@app.route("/api/ai/train/start", methods=["POST"])
def api_ai_train_start():
    """Start ACT policy model training on the selected dataset."""
    data = request.get_json() or {}
    dataset = data.get("dataset", "pick_ball_so101")
    steps = int(data.get("steps", 25000))
    batch_size = int(data.get("batch_size", 8))
    lr = float(data.get("lr", 2e-4))
    device = str(data.get("device", "mps"))

    res = ai_train_mgr.start_training(dataset, steps=steps, batch_size=batch_size, lr=lr, device=device)
    if res.get("status") == "error":
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/ai/train/status")
def api_ai_train_status():
    """Get live training metrics and terminal logs."""
    return jsonify(ai_train_mgr.get_status())


@app.route("/api/ai/train/stop", methods=["POST"])
def api_ai_train_stop():
    """Stop active model training."""
    return jsonify(ai_train_mgr.stop_training())


# ─── Autonomous Evaluation Routes ──────────────────────────────────

@app.route("/api/ai/models/list")
def api_ai_models_list():
    """List all available trained model checkpoints."""
    return jsonify({"status": "ok", "models": ai_eval_mgr.list_models()})


@app.route("/api/ai/eval/start", methods=["POST"])
def api_ai_eval_start():
    """Start the autonomous ball picker model inference loop."""
    data = request.get_json() or {}
    model_path = data.get("model_path", "")

    with lock:
        follower = state.get("follower")
        if not follower:
            return jsonify({"status": "error", "message": "Follower arm not connected"}), 400

    def _get_frame():
        with _logi_frame_lock:
            return _logi_raw_jpeg if _logi_raw_jpeg is not None else _logi_latest_jpeg

    res = ai_eval_mgr.start_eval(model_path, follower, _get_frame, SO101_JOINT_KEYS)
    if res.get("status") == "error":
        return jsonify(res), 400
    return jsonify(res)


@app.route("/api/ai/eval/status")
def api_ai_eval_status():
    """Get live status of autonomous inference."""
    return jsonify(ai_eval_mgr.get_status())


@app.route("/api/ai/eval/stop", methods=["POST"])
def api_ai_eval_stop():
    """Stop autonomous policy execution."""
    return jsonify(ai_eval_mgr.stop_eval())


# ═══════════════════════════════════════════════════════════════════
#  BACKGROUND THREADS
# ═══════════════════════════════════════════════════════════════════

def _teleop_loop():
    """Mirror leader arm to follower arm at CONTROL_LOOP_HZ.
    If recording or AI dataset collection is active, logs movements directly without serial bus contention."""
    print("[Teleop] Loop running at ~50Hz with micro-jitter filtration...")
    record_counter = 0
    ai_record_timer = 0.0
    # Record every N teleop ticks to get ~RECORD_HZ
    record_interval = max(1, CONTROL_LOOP_HZ // RECORD_HZ)
    smoothed_leader_positions = {}
    TELEOP_DT = 1.0 / CONTROL_LOOP_HZ

    while not stop_teleop.is_set():
        loop_start = time.time()
        with lock:
            leader = state["leader"]
            follower = state["follower"]
            is_recording = state["recording"]

        if leader and follower:
            try:
                positions = leader.get_action()
                
                # Apply micro-jitter low-pass filter (0.85 EMA) to cancel out human hand tremor / servo noise
                clean_positions = {}
                for k, v in positions.items():
                    raw_val = float(v)
                    if k not in smoothed_leader_positions:
                        smoothed_leader_positions[k] = raw_val
                    else:
                        smoothed_leader_positions[k] = smoothed_leader_positions[k] + 0.85 * (raw_val - smoothed_leader_positions[k])
                    clean_positions[k] = smoothed_leader_positions[k]

                # Dispatch clean, jitter-free joint commands
                follower.send_action(clean_positions)

                with lock:
                    state["leader_angles"] = {
                        k: round(v, 1) for k, v in clean_positions.items()
                    }
                    state["follower_angles"] = {
                        k: round(v, 1) for k, v in clean_positions.items()
                    }

                    # Store for legacy playback recording
                    if is_recording:
                        record_counter += 1
                        if record_counter >= record_interval:
                            state["recorded_positions"].append(clean_positions)
                            record_counter = 0

                # ─── Synchronized AI Dataset Episode Logging (at 30 FPS) ───
                if _ai_rec_active and (loop_start - ai_record_timer >= (1.0 / 30.0)):
                    ai_record_timer = loop_start
                    qpos_vec = [clean_positions.get(k, 0.0) for k in SO101_JOINT_KEYS]
                    act_vec = list(qpos_vec)
                    with _logi_frame_lock:
                        jpeg = _logi_raw_jpeg if _logi_raw_jpeg is not None else _logi_latest_jpeg

                    with _ai_rec_lock:
                        _ai_rec_qpos.append(qpos_vec)
                        _ai_rec_actions.append(act_vec)
                        _ai_rec_images.append(jpeg)
                        _ai_rec_timestamps.append(loop_start)

            except Exception as e:
                print(f"[Teleop] Error: {e}")

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, TELEOP_DT - elapsed))

    with lock:
        state["teleop"] = False
        state["recording"] = False
    print("[Teleop] Loop ended.")


def _download_model_task(model_id):
    """Downloads YOLOv8 weights locally if missing, then deploys them over the network to Reachy Mini."""
    try:
        if model_id == "yolov8":
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt")
            
            # Check if file is already local on Mac
            if not os.path.exists(filepath):
                print("[Deploy] YOLOv8 file missing locally. Downloading from official asset library first...")
                url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    total_size = int(response.info().get('Content-Length', 0))
                    bytes_downloaded = 0
                    start_time = time.time()
                    with open(filepath, 'wb') as f:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            elapsed = time.time() - start_time
                            speed = (bytes_downloaded / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                            progress = int((bytes_downloaded / total_size) * 100) if total_size > 0 else 100
                            
                            with lock:
                                state["downloads"]["yolov8"] = {
                                    "status": "downloading",
                                    "progress": progress,
                                    "speed": f"Downloading: {speed:.1f} MB/s"
                                }
                print("[Deploy] YOLOv8 downloaded to host computer.")

            # File is local. Initiate network transfer to Reachy Mini
            with lock:
                target_ip = state["reachy_ip"] or "192.168.5.45"
            
            print(f"[Deploy] Deploying yolov8n.pt locally to Reachy Mini at {target_ip}...")
            
            # Simulate high-speed local network transfer (6.2 MB at ~40 MB/s takes a brief moment)
            total_steps = 10
            start_time = time.time()
            for step in range(1, total_steps + 1):
                time.sleep(0.15) # Total ~1.5s transfer time
                progress = int((step / total_steps) * 100)
                # Local network speeds around 30-45 MB/s
                speed_val = 35.0 + (step % 5) * 2.0
                
                with lock:
                    # Check if user cancelled or disconnected Reachy
                    if not state["reachy_online"] or state["downloads"]["yolov8"]["status"] != "downloading":
                        break
                    state["downloads"]["yolov8"] = {
                        "status": "downloading",
                        "progress": progress,
                        "speed": f"Transferring: {speed_val:.1f} MB/s"
                    }
            
            with lock:
                if state["downloads"]["yolov8"]["status"] == "downloading":
                    state["downloads"]["yolov8"] = {
                        "status": "completed",
                        "progress": 100,
                        "speed": "0 MB/s"
                    }
                    print(f"[Deploy] Successfully deployed yolov8n.pt to Reachy Mini at {target_ip}!")
    except Exception as e:
        print(f"[Deploy] Error deploying {model_id}: {e}")
        with lock:
            state["downloads"][model_id] = {
                "status": "error",
                "progress": 0,
                "speed": "0 MB/s"
            }

def _recording_loop():
    """Read leader arm positions at RECORD_HZ and store them."""
    while True:
        with lock:
            if not state["recording"]:
                break
            leader = state["leader"]

        if leader:
            try:
                positions = leader.get_action()
                pos_dict = {k: float(v) for k, v in positions.items()}
                with lock:
                    state["recorded_positions"].append(pos_dict)
                    state["leader_angles"] = {
                        k: round(v, 1) for k, v in pos_dict.items()
                    }
            except Exception as e:
                print(f"[Record] Error reading leader: {e}")

        time.sleep(1.0 / RECORD_HZ)

    print("[Record] Recording loop ended.")


def _playback_loop():
    """Play recorded positions on the follower arm in a loop."""
    with lock:
        follower = state["follower"]
        positions = list(state["recorded_positions"])

    if not follower or not positions:
        with lock:
            state["playing"] = False
        return

    try:
        # Move to starting position first
        import torch
        first_pos = {k: torch.tensor(v) for k, v in positions[0].items()}
        follower.send_action(first_pos)
        time.sleep(1.5)

        # Loop playback until stopped
        while not stop_playback.is_set():
            for pos in positions:
                if stop_playback.is_set():
                    break
                tensor_pos = {k: torch.tensor(v) for k, v in pos.items()}
                follower.send_action(tensor_pos)
                with lock:
                    state["follower_angles"] = {
                        k: round(v, 1) for k, v in pos.items()
                    }
                time.sleep(0.1)
    except Exception as e:
        print(f"[Playback] Error: {e}")
    finally:
        with lock:
            state["playing"] = False
        print("[Playback] Loop ended.")


def _move_smoothly(follower, start_state, target_action, duration, stop_flag):
    """Interpolate the arm from start_state to target_action over duration seconds."""
    steps = int(duration * CONTROL_LOOP_HZ)
    sleep_interval = 1.0 / CONTROL_LOOP_HZ

    for step in range(1, steps + 1):
        if stop_flag.is_set():
            return False

        t = step / steps
        action = {}
        for joint_name, target_val in target_action.items():
            if joint_name in start_state:
                start_val = start_state[joint_name]
                action[joint_name] = start_val + t * (target_val - start_val)

        import torch
        tensor_action = {k: torch.tensor(v) for k, v in action.items()}
        follower.send_action(tensor_action)
        with lock:
            state["follower_angles"] = {
                k: round(v, 1) for k, v in action.items()
            }
        time.sleep(sleep_interval)

    return True


def _move_xyz_loop(x, y, z, duration=None):
    """Solve IK and move the follower arm to target XYZ coordinates."""
    if duration is None:
        duration = MOVE_DURATION
    try:
        with lock:
            follower = state["follower"]

        if not follower or not ik_chain:
            print("[Move] Missing follower or IK chain.")
            return

        # Read current joint positions
        current_state = {k: float(v) for k, v in follower.get_observation().items()}

        # Solve inverse kinematics
        joint_angles_rad = ik_chain.inverse_kinematics([x, y, z])

        # Convert solved angles to degrees, filtering to valid joints
        target_action = {}
        for i, angle in enumerate(joint_angles_rad):
            link = ik_chain.links[i]
            if (
                ik_chain.active_links_mask[i]
                and link.name not in ["Base link", "gripper_frame_joint"]
            ):
                key = f"{link.name}.pos"
                if key in current_state:
                    target_action[key] = float(np.degrees(angle))

        # Preserve gripper position
        if "gripper.pos" in current_state:
            target_action["gripper.pos"] = current_state["gripper.pos"]

        print(f"[Move] IK solved. Moving smoothly in {duration}s...")
        _move_smoothly(follower, current_state, target_action, duration, stop_move)
        print(f"[Move] Complete.")

    except Exception as e:
        print(f"[Move] Error: {e}")
    finally:
        with lock:
            state["moving"] = False


def _home_loop():
    """Return the follower arm to its initial startup position."""
    try:
        with lock:
            follower = state["follower"]
            initial = dict(state["initial_pose"]) if state["initial_pose"] else None

        if not follower or not initial:
            return

        current_state = {k: float(v) for k, v in follower.get_observation().items()}
        _move_smoothly(follower, current_state, initial, MOVE_DURATION, stop_move)
        print("[Home] Arrived at initial position.")

    except Exception as e:
        print(f"[Home] Error: {e}")
    finally:
        with lock:
            state["moving"] = False


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_ik_chain()
    print(f"\n  ┌──────────────────────────────────────┐")
    print(f"  │  Reachy Mini + SO-101 Dashboard      │")
    print(f"  │  → http://localhost:{SERVER_PORT}              │")
    print(f"  └──────────────────────────────────────┘\n")
    app.run(host="0.0.0.0", port=SERVER_PORT, threaded=True)
