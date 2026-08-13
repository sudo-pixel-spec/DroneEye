import os
import time
import json
import queue
import threading
import logging
from gcs.command_parser import CommandParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grammar word list passed to Vosk's KaldiRecognizer.
# Constraining the vocabulary to only command-relevant words dramatically
# improves recognition accuracy for a limited-command use-case.
# Add synonyms / new object names here as needed.
# ---------------------------------------------------------------------------
COMMAND_GRAMMAR = json.dumps([
    # --- actions ---
    "arm", "disarm", "takeoff", "land", "rtl", "return", "home",
    "hold", "loiter", "pause", "hover",
    "scan", "geo", "terrain", "environment",
    "track", "follow", "lock", "pursue", "chase", "keep",
    "search", "find", "locate", "detect", "identify", "spot", "look",
    # --- targets ---
    "person", "people", "human", "man", "woman", "pedestrian",
    "bottle", "flask", "canteen", "container",
    "backpack", "bag", "luggage", "suitcase",
    "car", "vehicle", "truck", "bus", "bicycle", "bike",
    "phone", "laptop", "book", "pen", "pencil", "notebook",
    # --- colors ---
    "red", "blue", "green", "yellow", "orange", "black", "white", "dark", "light",
    # --- numbers (for altitude) ---
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "fifteen", "twenty", "thirty",
    # --- fillers / polite words ---
    "the", "a", "an", "for", "with", "to", "up", "at", "on", "of",
    "please", "just", "now", "only", "go", "and", "can", "you",
    "i", "want", "let", "make", "that", "this", "hey", "ok", "okay",
    # unknown token — required by Vosk grammar mode
    "[unk]"
])


class VoiceEngine:
    """
    Offline-first voice recognition engine.

    Backend priority (RPi optimised):
      1. Vosk + sounddevice  — fully offline, ~5% CPU on RPi 5, 16kHz mono stream
      2. speech_recognition  — fallback, requires internet (Google STT)
      3. simulated           — no mic present; phrase input via API only

    Audio callback is kept MINIMAL: it only feeds audio to Vosk and puts
    final transcripts into a thread-safe queue.  A dedicated dispatch thread
    drains that queue and calls process_voice_phrase() — so exceptions from
    command parsing or the mission engine callback NEVER silently disappear
    inside the audio callback's try/except.

    Mic operates in TOGGLE mode: call toggle_listening() to start/stop capture.
    """

    def __init__(self, wake_word="jarvis", command_callback=None):
        self.wake_word = wake_word
        self.command_parser = CommandParser(wake_word=wake_word)
        self.command_callback = command_callback

        # Lifecycle
        self.running = False
        self._sr_thread = None
        self._cmd_thread = None

        # Thread-safe queue: audio callback → dispatch thread
        self._cmd_queue = queue.Queue()

        # Mic toggle state
        self.mic_active = False
        self._state_lock = threading.Lock()
        self._partial_text = ""
        self._last_transcript = ""
        self._last_action = ""

        # Vosk internals
        self._vosk_model = None
        self._recognizer = None
        self._rec_lock = threading.Lock()
        self._sd_stream = None
        self._sample_rate = 16000
        self._device_index = None

        # speech_recognition fallback internals
        self._sr_recognizer = None
        self._sr_microphone = None

        self.backend = "simulated"
        self._init_backend()

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_backend(self):
        if self._try_init_vosk():
            return
        if self._try_init_sr():
            return
        logger.info("[VoiceEngine] No mic backend available — simulated mode.")

    def _try_init_vosk(self):
        try:
            from vosk import Model, KaldiRecognizer
            import sounddevice  # noqa

            from config import GCS_CONFIG, MODELS_DIR

            raw_path = GCS_CONFIG.get("vosk_model_path", "vosk-model-small-en-us-0.15")
            model_path = raw_path if os.path.isabs(raw_path) else os.path.join(MODELS_DIR, raw_path)

            if not os.path.isdir(model_path):
                logger.warning(
                    f"[Vosk] Model not found: {model_path}\n"
                    "       Download: https://alphacephei.com/vosk/models (small-en-us ~40MB)\n"
                    "       Place in: models/vosk-model-small-en-us-0.15/"
                )
                return False

            self._sample_rate = int(GCS_CONFIG.get("vosk_sample_rate", 16000))
            self._device_index = GCS_CONFIG.get("voice_device_index", None)

            self._vosk_model = Model(model_path)
            self._recognizer = KaldiRecognizer(self._vosk_model, self._sample_rate, COMMAND_GRAMMAR)

            self.backend = "vosk"
            logger.info(f"[Vosk] Offline STT ready. Model: {os.path.basename(model_path)} | {self._sample_rate} Hz")
            return True

        except ImportError as exc:
            logger.warning(f"[Vosk] Not installed ({exc}). Run: pip install vosk sounddevice")
            return False
        except Exception as exc:
            logger.warning(f"[Vosk] Init failed: {exc}")
            return False

    def _try_init_sr(self):
        try:
            import speech_recognition as sr
            self._sr_recognizer = sr.Recognizer()
            self._sr_microphone = sr.Microphone()
            self.backend = "speech_recognition"
            logger.info("[SpeechRecognition] Fallback backend ready (requires internet).")
            return True
        except Exception as exc:
            logger.info(f"[SpeechRecognition] Not available: {exc}")
            return False

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self):
        if self.running:
            return
        self.running = True

        # Command dispatch thread — drains _cmd_queue, calls process_voice_phrase()
        # Runs independently of audio callback so exceptions propagate correctly
        self._cmd_thread = threading.Thread(
            target=self._cmd_dispatch_loop, daemon=True, name="VoiceCmdDispatch"
        )
        self._cmd_thread.start()

        if self.backend == "vosk":
            self._open_vosk_stream()
        elif self.backend == "speech_recognition":
            self._sr_thread = threading.Thread(
                target=self._sr_listen_loop, daemon=True, name="VoiceSR"
            )
            self._sr_thread.start()

        logger.info(f"[VoiceEngine] Started [{self.backend}] — mic IDLE, click 🎤 to activate")

    def stop(self):
        self.running = False
        self.mic_active = False
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None
        # Unblock cmd_dispatch_loop
        self._cmd_queue.put(None)
        if self._cmd_thread and self._cmd_thread.is_alive():
            self._cmd_thread.join(timeout=2.0)
        if self._sr_thread and self._sr_thread.is_alive():
            self._sr_thread.join(timeout=1.5)
        logger.info("[VoiceEngine] Stopped.")

    # ------------------------------------------------------------------
    # Command dispatch thread (runs separately from audio)
    # ------------------------------------------------------------------

    def _cmd_dispatch_loop(self):
        """
        Dedicated thread that drains the command queue.
        Separated from the audio callback so:
        - Audio callback stays lightweight (no heavy Python in real-time thread)
        - Exceptions from mission engine are fully logged, never silently lost
        """
        logger.info("[VoiceEngine] Command dispatch thread started.")
        while self.running:
            try:
                text = self._cmd_queue.get(timeout=0.5)
                if text is None:  # sentinel for shutdown
                    break
                self.process_voice_phrase(text)
            except queue.Empty:
                continue
            except Exception as exc:
                logger.error(f"[VoiceEngine] Dispatch error processing voice command: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    # Vosk: sounddevice InputStream — MINIMAL callback, just queues text
    # ------------------------------------------------------------------

    def _open_vosk_stream(self):
        import sounddevice as sd
        import numpy as np

        # ----------------------------------------------------------------
        # Query device's native rate/channels first.
        # On Windows WASAPI, requesting a non-native sample rate causes
        # the stream to open silently but deliver 0 samples.
        # Fix: capture at native rate, resample to 16kHz in the callback.
        # ----------------------------------------------------------------
        try:
            dev_info = sd.query_devices(self._device_index, kind='input') \
                if self._device_index is not None else sd.query_devices(kind='input')
            native_rate = int(dev_info.get('default_samplerate', self._sample_rate))
            native_ch   = max(1, min(2, int(dev_info.get('max_input_channels', 1))))
            dev_name    = dev_info.get('name', 'default')
        except Exception:
            native_rate = self._sample_rate
            native_ch   = 1
            dev_name    = 'default'

        vosk_rate = self._sample_rate  # 16000 Hz — fixed for Vosk model
        BLOCK     = int(native_rate * 0.5)  # 500ms block at native rate

        logger.info(
            f"[Vosk] Device: '{dev_name}' | {native_rate} Hz, {native_ch}ch "
            f"→ resampling to {vosk_rate} Hz mono for inference"
        )

        def _on_audio(indata, frames, time_info, status):
            """
            Audio I/O callback — intentionally minimal.
            1. Stereo → mono
            2. Resample native_rate → vosk_rate (16kHz)
            3. Feed bytes to Vosk
            4. Queue final transcript for dispatch thread
            """
            if not self.mic_active:
                return

            # --- Stereo → mono ---
            if native_ch > 1:
                mono = indata.reshape(-1, native_ch).mean(axis=1).astype(np.int16)
            else:
                mono = indata.flatten().astype(np.int16)

            # --- Resample to 16kHz (linear interp, fast, no extra deps) ---
            if native_rate != vosk_rate:
                n_in  = len(mono)
                n_out = int(n_in * vosk_rate / native_rate)
                mono  = np.interp(
                    np.linspace(0, n_in - 1, n_out),
                    np.arange(n_in),
                    mono.astype(np.float32)
                ).astype(np.int16)

            audio_bytes = mono.tobytes()

            with self._rec_lock:
                rec = self._recognizer
            if rec is None:
                return

            try:
                if rec.AcceptWaveform(audio_bytes):
                    result = json.loads(rec.Result())
                    text   = result.get("text", "").strip()
                    if text:
                        logger.info(f"[Vosk] ✓ Transcript: '{text}'")
                        with self._state_lock:
                            self._last_transcript = text
                            self._partial_text     = ""
                        self._cmd_queue.put(text)
                else:
                    partial = json.loads(rec.PartialResult())
                    p = partial.get("partial", "")
                    with self._state_lock:
                        self._partial_text = p
            except (ValueError, KeyError) as exc:
                logger.debug(f"[Vosk] Audio decode error: {exc}")

        try:
            self._sd_stream = sd.InputStream(
                samplerate=native_rate,
                blocksize=BLOCK,
                device=self._device_index,
                dtype='int16',
                channels=native_ch,
                callback=_on_audio,
            )
            self._sd_stream.start()
            logger.info("[Vosk] Audio input stream started ✓")

        except Exception as exc:
            logger.error(
                f"[Vosk] Failed to open audio stream: {exc}\n"
                f"       Run: python check_mic.py  to list available devices.\n"
                f"       Set 'voice_device_index' in config.py if needed."
            )
            self.backend = "simulated"

    # ------------------------------------------------------------------
    # speech_recognition fallback loop
    # ------------------------------------------------------------------

    def _sr_listen_loop(self):
        import speech_recognition as sr

        while self.running:
            if not self.mic_active:
                time.sleep(0.2)
                continue
            try:
                with self._sr_microphone as source:
                    self._sr_recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self._sr_recognizer.listen(source, timeout=3.0, phrase_time_limit=6.0)
                text = self._sr_recognizer.recognize_google(audio)
                logger.info(f"[SR] ✓ Transcript: '{text}'")
                with self._state_lock:
                    self._last_transcript = text
                    self._partial_text = ""
                # Also use the queue for consistency
                self._cmd_queue.put(text)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle_listening(self):
        """Toggle mic on/off. Resets Vosk buffer on every toggle."""
        self.mic_active = not self.mic_active

        if self.backend == "vosk" and self._vosk_model is not None:
            from vosk import KaldiRecognizer
            # Grammar-constrained recognizer for better accuracy
            new_rec = KaldiRecognizer(self._vosk_model, self._sample_rate, COMMAND_GRAMMAR)
            with self._rec_lock:
                self._recognizer = new_rec

        with self._state_lock:
            self._partial_text = ""

        logger.info(f"[VoiceEngine] Mic {'▶ ACTIVE — speak now' if self.mic_active else '■ IDLE'}")
        return self.mic_active

    def get_status(self):
        """Thread-safe status dict polled by the GCS web UI."""
        with self._state_lock:
            return {
                "backend":         self.backend,
                "listening":       self.mic_active,
                "partial":         self._partial_text,
                "last_transcript": self._last_transcript,
                "last_action":     self._last_action,
            }

    def process_voice_phrase(self, phrase_text):
        """
        Parse text and fire mission callback.
        Called from _cmd_dispatch_loop (dedicated thread) — NOT from audio callback.
        """
        logger.info(f"[VoiceEngine] Processing: '{phrase_text}'")
        cmd    = self.command_parser.parse_command(phrase_text)
        action = cmd.get("action", "UNKNOWN")

        if action != "UNKNOWN":
            logger.info(f"[VoiceEngine] ✅ Action={action} | Params={cmd.get('params', {})}")
            if self.command_callback:
                self.command_callback(cmd)
            with self._state_lock:
                self._last_action = f"Executed {action}"
        else:
            logger.warning(f"[VoiceEngine] ⚠ No command matched: '{phrase_text}'")
            with self._state_lock:
                self._last_action = "No match"

        return cmd
