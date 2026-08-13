import os
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

import cv2
cv2.setNumThreads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

CAMERA_CONFIG = {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "use_picamera2": True,
    "v4l2_device": 0,
    "enable_autofocus": True,
    "swap_bgr": False,
    "flip_v": False,
    "flip_h": False
}

VISION_CONFIG = {
    "model_name": "yolo11n_int8.onnx",
    "fallback_model_name": "yolo11n.onnx",
    "segmentation_model": "terrain_segmentation.onnx",
    "segmentation_input_size": (256, 256),
    "enable_ai_segmentation": True,
    "confidence_threshold": 0.35,
    "nms_threshold": 0.45,
    "input_size": (320, 320),
    "max_inference_fps": 12.0,
    "onnx_threads": 2
}

FLIGHT_CONFIG = {
    "connection_string": "/dev/ttyAMA0",
    "baud_rate": 57600,
    "sim_mode": True,
    "target_system": 1,
    "target_component": 1,
    "default_altitude": 10.0,
    "max_velocity_xy": 2.5,
    "max_accel_xy": 1.5,
    "pid_kp": 0.005,
    "pid_ki": 0.0001,
    "pid_kd": 0.001,
    "pid_yaw_kp": 0.05,
    "rc_override_channel": 5,
    "mav_frame": "MAV_FRAME_BODY_NED",
    "companion_heartbeat_interval": 1.0,
    "gcs_timeout_seconds": 5.0,
    "gps_stream_rate_hz": 10
}

QUALITY_PROFILES = {
    "low": {"width": 360, "height": 240, "quality": 40, "label": "LOW (360p / 40%)"},
    "balanced": {"width": 640, "height": 360, "quality": 65, "label": "BALANCED (480p / 65%)"},
    "hd": {"width": 854, "height": 480, "quality": 85, "label": "HD (720p / 85%)"}
}

GCS_CONFIG = {
    "host": "0.0.0.0",
    "web_port": 5000,
    "video_port": 8080,
    "wake_word": "jarvis",
    "stream_fps": 30,
    "default_quality_mode": "balanced"
}
