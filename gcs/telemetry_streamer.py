import time
import json
import threading
import logging
import cv2
from config import GCS_CONFIG, QUALITY_PROFILES

logger = logging.getLogger(__name__)

_current_jpeg_bytes = None
_current_telemetry_data = {}
_active_quality_mode = GCS_CONFIG.get("default_quality_mode", "balanced")
_stream_lock = threading.Lock()

def get_current_frame_jpeg():
    with _stream_lock:
        return _current_jpeg_bytes

def get_current_telemetry():
    with _stream_lock:
        return dict(_current_telemetry_data)

def set_quality_mode(mode_name):
    global _active_quality_mode
    mode_name = mode_name.lower().strip()
    if mode_name in QUALITY_PROFILES:
        with _stream_lock:
            _active_quality_mode = mode_name
        logger.info(f"Dynamic Video Quality switched to [{mode_name.upper()}]: {QUALITY_PROFILES[mode_name]['label']}")
        return QUALITY_PROFILES[mode_name]
    return None

def get_active_quality_mode():
    with _stream_lock:
        return _active_quality_mode

class TelemetryStreamer:
    def __init__(self, host=None, port=None):
        self.host = host or GCS_CONFIG.get("host", "0.0.0.0")
        self.port = port or GCS_CONFIG.get("video_port", 8080)
        self.running = False
        self._latest_raw_frame = None
        self._frame_lock = threading.Lock()
        self._encoder_thread = None

    def start(self):
        self.running = True
        self._encoder_thread = threading.Thread(target=self._async_encoding_loop, daemon=True)
        self._encoder_thread.start()
        logger.info("Video & Telemetry Engine initialized with Async Background JPEG Encoding.")

    def update_frame(self, frame):
        if frame is None:
            return
        with self._frame_lock:
            self._latest_raw_frame = frame

    def _async_encoding_loop(self):
        global _current_jpeg_bytes
        target_interval = 1.0 / 25.0
        
        while self.running:
            start_t = time.time()
            frame_to_encode = None

            with self._frame_lock:
                if self._latest_raw_frame is not None:
                    frame_to_encode = self._latest_raw_frame
                    self._latest_raw_frame = None

            if frame_to_encode is not None:
                try:
                    with _stream_lock:
                        q_mode = _active_quality_mode

                    profile = QUALITY_PROFILES.get(q_mode, QUALITY_PROFILES["balanced"])
                    target_w = profile["width"]
                    target_h = profile["height"]
                    target_q = profile["quality"]

                    h, w = frame_to_encode.shape[:2]
                    if w != target_w or h != target_h:
                        frame_out = cv2.resize(frame_to_encode, (target_w, target_h),
                                               interpolation=cv2.INTER_NEAREST if q_mode == "low" else cv2.INTER_LINEAR)
                    else:
                        frame_out = frame_to_encode

                    ret, jpeg = cv2.imencode(".jpg", frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), target_q])
                    if ret:
                        jpeg_bytes = jpeg.tobytes()
                        with _stream_lock:
                            _current_jpeg_bytes = jpeg_bytes
                except Exception as e:
                    logger.error(f"Async JPEG encoder error: {e}")

            elapsed = time.time() - start_t
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def update_telemetry(self, telemetry_dict):
        global _current_telemetry_data
        with _stream_lock:
            telemetry_dict["quality_mode"] = _active_quality_mode.upper()
            _current_telemetry_data = telemetry_dict

    def stop(self):
        self.running = False
        if self._encoder_thread and self._encoder_thread.is_alive():
            self._encoder_thread.join(timeout=1.0)
        logger.info("Telemetry streamer stopped.")
