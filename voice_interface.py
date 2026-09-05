import os
import sys
import time
import threading
import subprocess
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import imageio_ffmpeg
from dotenv import load_dotenv

# Ensure ffmpeg binary is on PATH for Whisper
try:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    symlink_path = os.path.join(ffmpeg_dir, "ffmpeg")
    if not os.path.exists(symlink_path):
        try:
            os.symlink(ffmpeg_exe, symlink_path)
        except Exception:
            pass
    bin_dir = os.path.expanduser("~/miniforge3/envs/lerobot/bin")
    os.environ["PATH"] = f"{bin_dir}:{ffmpeg_dir}:" + os.environ.get("PATH", "")
except Exception as e:
    print(f"[Voice] Warning setting up FFmpeg PATH: {e}")

import whisper
from google import genai
from google.genai import types
from reachy_mini import ReachyMini

# Load API keys
load_dotenv(os.path.expanduser("~/.env"))
load_dotenv()

# Constants
ROBOT_IP = "192.168.5.45"
RATE = 16000
CHANNELS = 1
temp_wav = "scratch/temp_voice.wav"
speech_aiff = "scratch/speech.aiff"
speech_wav = "scratch/speech.wav"

# Global Reachy Client
mini = None

def get_current_frame():
    """Fetches the best available camera frame.
    Priority: 1) Logitech HD webcam via dashboard snapshot endpoint
              2) Reachy Mini's WebRTC camera (fallback)
    """
    # Try Logitech webcam first (higher quality, 1080p, wider FOV)
    try:
        import urllib.request
        import numpy as np
        import cv2
        with urllib.request.urlopen('http://localhost:8080/api/logi_snapshot', timeout=2) as r:
            data = r.read()
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    except Exception:
        pass
    # Fallback to Reachy Mini's camera
    if mini and hasattr(mini, 'media'):
        return mini.media.get_frame()
    return None

def move_head(yaw: float, pitch: float) -> str:
    """Commands Reachy Mini's neck actuators (pan/tilt).
    yaw: Left/Right angle in degrees (range -60 to 60). Negative is right, positive is left.
    pitch: Up/Down angle in degrees (range -30 to 30). Negative is down, positive is up.
    """
    if not mini:
        return "Robot client not initialized."
    try:
        from reachy_mini.utils import create_head_pose
        pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
        mini.goto_target(head=pose, duration=0.8)
        return f"Moved head to yaw={yaw}°, pitch={pitch}°."
    except Exception as e:
        return f"Failed to move head: {e}"

def move_antennas(left_angle: float, right_angle: float) -> str:
    """Moves Reachy Mini's left and right antennas independently.
    left_angle: Left antenna angle in degrees (range -90 to 90).
    right_angle: Right antenna angle in degrees (range -90 to 90).
    """
    if not mini:
        return "Robot client not initialized."
    try:
        left_rad = np.deg2rad(left_angle)
        right_rad = np.deg2rad(right_angle)
        mini.goto_target(antennas=[left_rad, right_rad], duration=0.5)
        return f"Moved antennas to left={left_angle}°, right={right_angle}°."
    except Exception as e:
        return f"Failed to move antennas: {e}"

def move_torso(yaw: float) -> str:
    """Rotates Reachy Mini's body/torso left or right.
    yaw: Body rotation angle in degrees (range -45 to 45). Positive is turn left, negative is turn right.
    """
    if not mini:
        return "Robot client not initialized."
    try:
        body_yaw_rad = np.deg2rad(yaw)
        mini.goto_target(body_yaw=body_yaw_rad, duration=1.0)
        return f"Rotated torso to yaw={yaw}°."
    except Exception as e:
        return f"Failed to move torso: {e}"

def play_gesture(gesture: str) -> str:
    """Performs an expressive robotic gesture animation.
    gesture: 'yes', 'no', 'excited', 'sad', 'confused', or 'look_around'.
    """
    if not mini:
        return "Robot client not initialized."
    try:
        from reachy_mini.utils import create_head_pose
        if gesture == "yes":
            for pitch in [15, -10, 15, -10, 0]:
                pose = create_head_pose(yaw=0, pitch=pitch, degrees=True)
                mini.goto_target(head=pose, duration=0.25)
                time.sleep(0.25)
        elif gesture == "no":
            for yaw in [-20, 20, -20, 20, 0]:
                pose = create_head_pose(yaw=yaw, pitch=0, degrees=True)
                mini.goto_target(head=pose, duration=0.25)
                time.sleep(0.25)
        elif gesture == "excited":
            for i in range(3):
                pose = create_head_pose(yaw=0, pitch=15 if i % 2 == 0 else -10, degrees=True)
                mini.goto_target(
                    head=pose,
                    antennas=np.deg2rad([90, -90] if i % 2 == 0 else [-90, 90]),
                    duration=0.2
                )
                time.sleep(0.2)
            pose = create_head_pose(yaw=0, pitch=0, degrees=True)
            mini.goto_target(head=pose, antennas=np.deg2rad([0, 0]), duration=0.3)
        elif gesture == "sad":
            pose = create_head_pose(yaw=0, pitch=-20, degrees=True)
            mini.goto_target(head=pose, antennas=np.deg2rad([-45, 45]), duration=1.0)
        elif gesture == "confused":
            pose = create_head_pose(yaw=15, pitch=10, degrees=True)
            mini.goto_target(head=pose, antennas=np.deg2rad([90, -30]), duration=0.8)
        elif gesture == "look_around":
            for yaw, torso_yaw in [(-30, -30), (30, 30), (0, 0)]:
                pose = create_head_pose(yaw=yaw, pitch=5, degrees=True)
                mini.goto_target(head=pose, body_yaw=np.deg2rad(torso_yaw), duration=1.2)
                time.sleep(1.3)
        return f"Executed gesture '{gesture}'."
    except Exception as e:
        return f"Gesture error: {e}"

def ask_assistant(prompt: str) -> str:
    """Forwards a coding, configuration, scripting, or UI modification request to the Antigravity AI coding assistant to execute in this workspace.
    prompt: The detailed action or command for the coding assistant.
    """
    try:
        os.makedirs("scratch", exist_ok=True)
        with open("scratch/assistant_prompt.txt", "w") as f:
            f.write(prompt)
        print(f"\n[Bridge] Forwarded prompt to coding assistant: '{prompt}'")
        return f"Successfully forwarded request to the coding assistant: '{prompt}'"
    except Exception as e:
        return f"Failed to forward request to assistant: {e}"

# Pre-load YOLO model once globally to avoid reloading latency
yolo_model = None
try:
    from ultralytics import YOLO
    print("[Vision] Pre-loading YOLOv8s object detection model...")
    yolo_model = YOLO("yolov8s.pt")
    print("[Vision] YOLOv8s model ready!")
except Exception as _e:
    print(f"[Vision] Note: YOLO pre-load warning: {_e}")

def run_yolo_detection(frame) -> str:
    """Fast YOLOv8 object detection fallback."""
    global yolo_model
    try:
        if yolo_model is None:
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8s.pt")
            
        results = yolo_model(frame, verbose=False)
        detected = []
        for r in results:
            for c in r.boxes.cls:
                name = yolo_model.names[int(c)]
                detected.append(name)
                
        if not detected:
            return "I do not see any recognized objects in front of me right now."
            
        counts = {}
        for item in detected:
            counts[item] = counts.get(item, 0) + 1
            
        items_str = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" for name, count in counts.items()])
        return f"I see the following objects: {items_str}."
    except Exception as e:
        return f"Object detection error: {e}"

def identify_objects(prompt: str = None) -> str:
    """Captures a frame from the Logitech HD webcam (primary) or Reachy Mini's camera (fallback)
    and uses Gemini Vision VLM + YOLO to identify and describe objects, text, colors, and scene details.
    prompt: Optional specific visual query (e.g. 'Is there a red cup on the table?', 'What text is on that paper?').
    """
    print("👁️ [Vision Tool] Capturing camera frame & running AI Multimodal Vision...")
    frame = get_current_frame()
    if frame is None:
        return "Failed to grab frame from any available camera."
        
    try:
        import cv2
        import PIL.Image
        
        # Convert BGR frame from OpenCV to RGB PIL Image for Gemini VLM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb_frame)
        
        vlm_prompt = prompt if (prompt and prompt.strip()) else (
            "You are Reachy Mini looking through your camera. Describe what you see in front of you clearly and concisely. "
            "Mention the main objects, their colors, relative spatial positions (left/right/center), "
            "and any readable text. Keep response under 3 sentences for speaking out loud."
        )
        
        vlm_client = genai.Client()
        response = vlm_client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[pil_img, vlm_prompt]
        )
        
        if response and response.text:
            res_text = response.text.strip()
            print(f"   └─ Vision Result: {res_text}")
            return res_text
        else:
            return run_yolo_detection(frame)
    except Exception as e:
        print(f"   └─ Gemini VLM fallback to YOLO due to: {e}")
        return run_yolo_detection(frame)

def control_arms(action: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> str:
    """Controls the separate SO-101 robotic arms on the desk.
    action: 'home', 'move', 'teleop', 'record', or 'replay'.
    x, y, z: Position offsets in meters when action='move'.
    """
    try:
        res = requests.post("http://localhost:8080/api/arms/control", json={"action": action, "x": x, "y": y, "z": z}, timeout=3)
        if res.status_code == 200:
            return f"Arm action '{action}' executed successfully."
        return f"Arm server returned status {res.status_code}."
    except Exception as e:
        return f"Arm control failed: {e}"


def list_detected_objects() -> str:
    """Returns a list of all objects currently detected on the table by the overhead camera.
    Use this when the user asks 'what's on the table?', 'what do you see on the desk?', or similar.
    """
    print("📦 [Vision Tool] Fetching detected objects from table camera...")
    try:
        res = requests.get("http://localhost:8080/api/detected_objects", timeout=3)
        data = res.json()
        objects = data.get("objects", [])
        if not objects:
            return "No objects are currently detected on the table. Make sure AI Vision is enabled in the dashboard."
        lines = []
        for obj in objects:
            name = obj.get("name", "unknown")
            conf = int(obj.get("confidence", 0) * 100)
            if obj.get("calibrated") and obj.get("robot_x") is not None:
                rx, ry = obj["robot_x"], obj["robot_y"]
                lines.append(f"{name} ({conf}% confidence, at X={rx:.3f}m Y={ry:.3f}m on table)")
            else:
                lines.append(f"{name} ({conf}% confidence, position not calibrated)")
        result = "Objects detected on the table: " + "; ".join(lines) + "."
        print(f"   └─ {result}")
        return result
    except Exception as e:
        return f"Could not fetch detected objects: {e}"


def go_to_object(object_name: str, z_offset: float = 0.0) -> str:
    """Moves the robot arm to a named object detected on the table by the overhead camera.
    object_name: The name or description of the object to move to (e.g. 'ball', 'orange ball', 'cup', 'screw').
    z_offset: Extra height above table in meters (default 0 = touch table). Use 0.05 to hover above it.
    The system will fuzzy-match the name against what the camera currently sees.
    """
    print(f"🎯 [Voice→Arm] Searching for '{object_name}' on the table...")
    try:
        res = requests.get("http://localhost:8080/api/detected_objects", timeout=3)
        data = res.json()
        objects = data.get("objects", [])
    except Exception as e:
        return f"Could not reach the vision system: {e}"

    if not objects:
        return "No objects are detected on the table right now. Make sure AI Vision is enabled in the dashboard."

    # Fuzzy match: score each detected object against the spoken name
    query = object_name.lower().strip()
    best_obj = None
    best_score = 0
    for obj in objects:
        if not obj.get("calibrated") or obj.get("robot_x") is None:
            continue  # Skip uncalibrated detections
        label = obj.get("name", "").lower()
        # Count how many words from the query appear in the label
        query_words = query.split()
        score = sum(1 for w in query_words if w in label)
        # Also reward exact substring match
        if query in label or label in query:
            score += 3
        if score > best_score:
            best_score = score
            best_obj = obj

    if best_obj is None or best_score == 0:
        names = [o.get("name", "?") for o in objects if o.get("calibrated")]
        return (f"Could not find '{object_name}' on the table. "
                f"Currently detected: {', '.join(names) if names else 'nothing calibrated'}.")

    rx = best_obj["robot_x"]
    ry = best_obj["robot_y"]
    name = best_obj["name"]
    conf = int(best_obj.get("confidence", 0) * 100)
    # Table Z height — tuned to touch the table surface
    z = 0.0048 + z_offset

    print(f"   └─ Matched '{name}' ({conf}%) at robot X={rx:.3f} Y={ry:.3f} → moving arm...")
    try:
        move_res = requests.post(
            "http://localhost:8080/api/move_xyz",
            json={"x": rx, "y": ry, "z": z, "duration": 3.0},
            timeout=5
        )
        if move_res.status_code == 200:
            return (f"Moving arm to the {name} at X={rx:.3f}m, Y={ry:.3f}m on the table.")
        else:
            detail = move_res.json().get("message", move_res.text)
            return f"Arm move failed: {detail}"
    except Exception as e:
        return f"Could not send move command: {e}"


def home_arm() -> str:
    """Sends the robot arm back to its home/resting position.
    Use when the user says 'go home', 'home the arm', 'return to home', 'reset the arm', etc.
    """
    print("🏠 [Arm] Sending arm to home position...")
    try:
        res = requests.post("http://localhost:8080/api/home", timeout=5)
        if res.status_code == 200:
            return "Arm is returning to home position."
        detail = res.json().get("message", res.text)
        return f"Home failed: {detail}"
    except Exception as e:
        return f"Could not reach the arm server: {e}"


def get_current_time() -> str:
    """Returns the current local time, day of the week, and date."""
    print("⏰ [Clock Tool] Querying system clock...")
    now = time.localtime()
    time_str = time.strftime("%I:%M %p", now)
    date_str = time.strftime("%A, %B %d, %Y", now)
    res = f"The current time is {time_str} on {date_str}."
    print(f"   └─ Clock Result: {res}")
    return res

def get_weather(location: str = None) -> str:
    """Fetches real-time weather information and forecast for a specified city or auto-detected location."""
    loc_display = location or "local area"
    print(f"🌤️ [Weather Tool] Fetching live weather for '{loc_display}'...")
    try:
        target_loc = location.strip() if location and location.strip() else ""
        url = f"https://wttr.in/{target_loc}?format=j1" if target_loc else "https://wttr.in/?format=j1"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            current = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]
            city = area.get("areaName", [{}])[0].get("value", "your area")
            region = area.get("region", [{}])[0].get("value", "")
            temp_f = current.get("temp_F", "N/A")
            desc = current.get("weatherDesc", [{}])[0].get("value", "").lower()
            feels_f = current.get("FeelsLikeF", "N/A")
            out = f"In {city}{', ' + region if region else ''}, it is currently {desc} and {temp_f}°F (feels like {feels_f}°F)."
            print(f"   └─ Weather Result: {out}")
            return out
        out = f"Could not fetch weather data for '{location or 'current location'}'."
        print(f"   └─ Weather Result: {out}")
        return out
    except Exception as e:
        out = f"Weather lookup error: {e}"
        print(f"   └─ Weather Result: {out}")
        return out

def set_timer(duration_seconds: int, timer_name: str = "Timer") -> str:
    """Sets a background timer countdown that alerts when finished."""
    print(f"⏱️ [Timer Tool] Setting countdown '{timer_name}' for {duration_seconds}s...")
    if duration_seconds <= 0:
        return "Please specify a positive duration in seconds for the timer."
        
    def _timer_worker():
        time.sleep(duration_seconds)
        alert_msg = f"Timer '{timer_name}' for {duration_seconds} seconds has finished!"
        print(f"\n⏰ [ALARM] {alert_msg}")
        speak(alert_msg)
        
    threading.Thread(target=_timer_worker, daemon=True).start()
    mins = duration_seconds // 60
    secs = duration_seconds % 60
    time_fmt = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
    res = f"Timer '{timer_name}' set for {time_fmt}."
    print(f"   └─ Timer Result: {res}")
    return res

def speak(text: str):
    """Speaks out loud using Reachy Mini's onboard physical speaker or system TTS."""
    print(f"\n[Reachy Mini] {text}")
    
    if mini and hasattr(mini, 'media'):
        try:
            os.makedirs("scratch", exist_ok=True)
            if os.path.exists(speech_wav):
                os.remove(speech_wav)
                
            subprocess.run([
                "say", "-o", speech_wav, 
                "--data-format=LEI16@16000", 
                text
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            mini.media.play_sound(speech_wav)
            words = len(text.split())
            duration = max(0.4, (words / 180.0) * 60.0 + 0.1)
            time.sleep(duration)
            return
        except Exception as e:
            print(f"[Speech Error] {e}")

    subprocess.run(["say", text])

def calibrate_threshold() -> float:
    """Calibrates baseline silence/noise threshold from the environment."""
    print("[Voice] Calibrating microphone ambient noise levels... Please stay quiet.")
    frames = []
    
    if mini and hasattr(mini, 'media'):
        try:
            start = time.time()
            while time.time() - start < 1.5:
                sample = mini.media.get_audio_sample()
                if sample is not None and len(sample) > 0:
                    pcm_16 = (sample * 32767.0).astype(np.int16)
                    frames.append(pcm_16)
                time.sleep(0.02)
            if frames:
                data = np.concatenate(frames, axis=0).astype(np.float64)
                baseline = np.sqrt(np.mean(data**2))
                threshold = max(300.0, baseline * 1.8)
                print(f"[Voice] Reachy Mic Calibration complete. Baseline: {baseline:.1f}, Threshold: {threshold:.1f}")
                return threshold
        except Exception:
            pass

    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16') as stream:
            for _ in range(int(RATE / 1024 * 1.5)):
                data, _ = stream.read(1024)
                frames.append(data)
        data = np.concatenate(frames, axis=0).astype(np.float64)
        baseline = np.sqrt(np.mean(data**2))
        threshold = max(300.0, baseline * 1.8)
        print(f"[Voice] Host Mic Calibration complete. Baseline: {baseline:.1f}, Threshold: {threshold:.1f}")
        return threshold
    except Exception:
        return 350.0

def record_until_silence(threshold: float) -> np.ndarray:
    """Listens continuously from Reachy Mini's onboard microphone or system mic until silence."""
    print("\n[Voice] Listening... (Start speaking when ready)")
    audio_frames = []
    
    if mini and hasattr(mini, 'media'):
        try:
            silence_samples = 0
            is_speaking = False
            start_time = time.time()
            
            while True:
                sample = mini.media.get_audio_sample()
                if sample is not None and len(sample) > 0:
                    pcm_16 = (sample * 32767.0).astype(np.int16)
                    rms = np.sqrt(np.mean(pcm_16.astype(np.float64)**2))
                    
                    if not is_speaking:
                        if rms > threshold:
                            sys.stdout.write("[Voice] Recording...")
                            sys.stdout.flush()
                            is_speaking = True
                            audio_frames.append(pcm_16)
                            silence_samples = 0
                    else:
                        audio_frames.append(pcm_16)
                        if rms < threshold:
                            silence_samples += len(pcm_16)
                            if silence_samples >= RATE * 0.7:
                                sys.stdout.write(" Done.\n")
                                sys.stdout.flush()
                                break
                        else:
                            silence_samples = 0
                else:
                    time.sleep(0.01)
                
                if time.time() - start_time > 20.0:
                    break
                    
            if audio_frames:
                return np.concatenate(audio_frames, axis=0)
        except Exception:
            pass

    try:
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, dtype='int16') as stream:
            for _ in range(2): stream.read(1024)
            is_speaking = False
            silence_chunks = 0
            silence_limit = int(RATE / 1024 * 0.7)
            while True:
                data, _ = stream.read(1024)
                rms = np.sqrt(np.mean(data.astype(np.float64)**2))
                if not is_speaking:
                    if rms > threshold:
                        sys.stdout.write("[Voice] Recording...")
                        sys.stdout.flush()
                        is_speaking = True
                        audio_frames.append(data)
                        silence_chunks = 0
                else:
                    audio_frames.append(data)
                    if rms < threshold:
                        silence_chunks += 1
                        if silence_chunks >= silence_limit:
                            sys.stdout.write(" Done.\n")
                            sys.stdout.flush()
                            break
                    else:
                        silence_chunks = 0
        if audio_frames:
            return np.concatenate(audio_frames, axis=0)
    except Exception:
        pass
        
    return np.zeros(RATE * 2, dtype=np.int16)

def main():
    global mini
    os.makedirs("scratch", exist_ok=True)
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("[Error] GEMINI_API_KEY is missing from environment. Please add it first.")
        sys.exit(1)
        
    print(f"[Robot] Connecting to Reachy Mini at {ROBOT_IP}...")
    try:
        mini = ReachyMini(host=ROBOT_IP, connection_mode="network", media_backend="webrtc")
        mini.__enter__()
        print("[Robot] Direct WebRTC Connection active!")
    except Exception as e:
        print(f"[Robot] Connection failed: {e}. Running in voice-only debug mode.")
        mini = None

    print("[Voice] Loading fast English Whisper model 'base.en'...")
    whisper_model = whisper.load_model("base.en")
    print("[Voice] Fast Whisper model loaded!")

    threshold = calibrate_threshold()

    print("[Gemini] Initializing chat session...")
    ai_client = genai.Client()
    
    system_instruction = (
        "You are Reachy Mini, a friendly and highly expressive robotic companion sitting on Nam's desk. "
        "You can speak, move your head (yaw/pan, pitch/tilt), wiggle your antennas, rotate your torso (body rotation), and talk to your "
        "creator's coding assistant (Antigravity) on screen. "
        "You do not have built-in robotic arms, but you can control the separate SO-101 robotic arms "
        "on the desk by calling the tool `control_arms(action, x, y, z)`. "
        "Whenever the user asks you to do something with the arms (e.g. home the arms, move the arms, "
        "mirror/teleoperate the arms, record a motion, or replay a motion), you MUST call the `control_arms` tool. "
        "Whenever the user asks to change code, write a script, edit the dashboard UI, or asks you to "
        "'tell the assistant to [do something]' or 'ask the assistant to [do something]', you MUST call the "
        "tool `ask_assistant(prompt)` with the user's detailed request. "
        "Whenever the user asks you to look somewhere (e.g. look left, look up) or express emotions "
        "(e.g. look sad, look happy, show excitement), you should call the appropriate movement tool (move_head, move_antennas, or move_torso). "
        "Whenever the user asks you to turn or rotate your body/torso (e.g. turn left, face right, rotate torso), you MUST call the tool `move_torso(yaw)`. "
        "Whenever you want to express a complex emotion or perform a gesture (e.g., nod 'yes', shake your head 'no', look excited, look sad, look confused, or look around to scan the room), you MUST call the tool `play_gesture(gesture)`. "
        "Whenever the user asks you 'what do you see?', 'what's in front of you?', or asks you to identify objects, "
        "you MUST call the tool `identify_objects()` to run live object detection and describe what you see. "
        "Whenever the user asks for the current time, date, or day of the week, you MUST call the tool `get_current_time()` to check the system clock. "
        "Whenever the user asks for the weather, forecast, or temperature, you MUST call the tool `get_weather(location)` to fetch the local forecast. Pass the location name if the user specifies a city (e.g. 'Bellevue', 'London'), or leave it empty to auto-detect."
        "Whenever the user asks you to set a timer, countdown, or alarm, you MUST call the tool `set_timer(duration_seconds, timer_name)` to configure it. "
        "Whenever the user asks you to move the arm to an object on the table (e.g. 'go to the ball', 'pick up the orange ball', "
        "'move to the cup', 'point at the screw'), you MUST call the tool `go_to_object(object_name)` with the object name. "
        "Whenever the user asks 'what's on the table?', 'what objects do you see on the desk?', or anything about listing table objects, "
        "you MUST call `list_detected_objects()` (NOT `identify_objects()`) since it uses the overhead camera's AI detection. "
        "Whenever the user says 'go home', 'home the arm', 'return to home', or 'reset the arm', you MUST call the tool `home_arm()`. "
        "Keep your spoken responses short and conversational (1-2 sentences maximum), as they will be spoken out loud."
    )
    
    chat = ai_client.chats.create(
        model="gemini-3.1-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[move_head, move_antennas, move_torso, play_gesture, ask_assistant, identify_objects, control_arms, get_current_time, get_weather, set_timer, go_to_object, list_detected_objects, home_arm],
        )
    )

    welcome_text = "Hello! Reachy Mini is online. What would you like to do today?"
    speak(welcome_text)

    try:
        while True:
            audio_data = record_until_silence(threshold)
            duration = len(audio_data) / RATE
            if duration < 0.6:
                continue
                
            sf.write(temp_wav, audio_data, RATE)
            
            print("[Voice] Transcribing...")
            try:
                result = whisper_model.transcribe(temp_wav, fp16=False)
                text = result["text"].strip()
            except Exception as e:
                print(f"[Voice] Transcription error: {e}")
                continue
            
            if not text or len(text) < 2:
                print("[Voice] Could not understand speech.")
                continue
                
            print(f"\n[You] {text}")
            
            print("[Gemini] Thinking...")
            try:
                response = chat.send_message(text)
                
                if response.function_calls:
                    for call in response.function_calls:
                        name = call.name
                        args = dict(call.args) if call.args else {}
                        print(f"⚡ [Function Call] Gemini triggered tool: {name}({args})")
                        
                        if name == "move_head":
                            result = move_head(yaw=float(args.get("yaw", 0)), pitch=float(args.get("pitch", 0)))
                        elif name == "move_antennas":
                            result = move_antennas(left_angle=float(args.get("left_angle", 0)), right_angle=float(args.get("right_angle", 0)))
                        elif name == "move_torso":
                            result = move_torso(yaw=float(args.get("yaw", 0.0)))
                        elif name == "play_gesture":
                            result = play_gesture(gesture=str(args.get("gesture", "")))
                        elif name == "ask_assistant":
                            result = ask_assistant(prompt=str(args.get("prompt", "")))
                        elif name == "identify_objects":
                            result = identify_objects()
                        elif name == "control_arms":
                            result = control_arms(
                                action=str(args.get("action", "")),
                                x=float(args.get("x", 0.0)),
                                y=float(args.get("y", 0.0)),
                                z=float(args.get("z", 0.0))
                            )
                        elif name == "get_current_time":
                            result = get_current_time()
                        elif name == "get_weather":
                            result = get_weather(
                                location=args.get("location", None)
                            )
                        elif name == "set_timer":
                            result = set_timer(
                                duration_seconds=int(args.get("duration_seconds", 0)),
                                timer_name=str(args.get("timer_name", "Timer"))
                            )
                        elif name == "go_to_object":
                            result = go_to_object(
                                object_name=str(args.get("object_name", "")),
                                z_offset=float(args.get("z_offset", 0.0))
                            )
                        elif name == "list_detected_objects":
                            result = list_detected_objects()
                        elif name == "home_arm":
                            result = home_arm()
                        else:
                            result = "Unknown function"
                        
                        print(f"📋 [Function Result] -> {result}")
                        response = chat.send_message(
                            types.Part.from_function_response(
                                name=name,
                                response={"result": result}
                            )
                        )
                
                response_text = response.text
                speak(response_text)
                
            except Exception as e:
                print(f"[Gemini] Error: {e}")
                
    finally:
        if mini:
            print("[Robot] Closing connection...")
            try:
                mini.__exit__(None, None, None)
            except Exception:
                pass
        try:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
            if os.path.exists(speech_aiff):
                os.remove(speech_aiff)
            if os.path.exists(speech_wav):
                os.remove(speech_wav)
        except Exception:
            pass

if __name__ == "__main__":
    main()
