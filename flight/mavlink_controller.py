import time
import math
import threading
import logging
import sys
import os

user_site = os.path.expanduser("~/.local/lib/python3.13/site-packages")
if user_site not in sys.path and os.path.exists(user_site):
    sys.path.insert(0, user_site)

from config import FLIGHT_CONFIG

logger = logging.getLogger(__name__)

MAV_MODE_GUIDED = "GUIDED"
MAV_MODE_RTL = "RTL"
MAV_MODE_LAND = "LAND"
MAV_MODE_LOITER = "LOITER"
MAV_MODE_AUTO = "AUTO"

class MavlinkController:
    def __init__(self, config=None):
        self.config = config or FLIGHT_CONFIG
        self.connection_string = self.config.get("connection_string", "/dev/ttyAMA0")
        self.baud_rate = self.config.get("baud_rate", 57600)
        self.sim_mode = self.config.get("sim_mode", True)
        
        self.master = None
        self.connected = False
        self.simulated = False
        
        self.state = {
            "armed": False,
            "mode": "DISCONNECTED",
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "heading": 0.0,
            "gps_fix": 0,
            "satellites_visible": 0,
            "battery_pct": 100,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "manual_override": False,
            "gcs_connected": True,
            "target_gps_lat": None,
            "target_gps_lon": None
        }
        
        self.pid_kp = self.config.get("pid_kp", 0.005)
        self.pid_ki = self.config.get("pid_ki", 0.0001)
        self.pid_kd = self.config.get("pid_kd", 0.001)
        self.pid_yaw_kp = self.config.get("pid_yaw_kp", 0.05)
        self.max_vel = self.config.get("max_velocity_xy", 2.5)
        self.max_accel = self.config.get("max_accel_xy", 1.5)
        self.mav_frame_str = self.config.get("mav_frame", "MAV_FRAME_BODY_NED")
        self.gcs_timeout = self.config.get("gcs_timeout_seconds", 5.0)
        
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.last_pid_time = time.time()
        self.last_heartbeat_sent = 0.0
        self.last_gcs_ping = time.time()
        
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

        self.connect()

    def connect(self):
        try:
            from pymavlink import mavutil
            logger.info(f"Attempting MAVLink serial connection to Pixhawk Telemetry 2 on {self.connection_string} @ {self.baud_rate}...")
            self.master = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baud_rate,
                autoreconnect=True,
                source_system=255,
                source_component=190
            )

            hb = self.master.wait_heartbeat(timeout=2.5)
            if hb:
                self.connected = True
                self.simulated = False
                self.state["mode"] = "INITIALIZED"
                logger.info(f"Connected to Pixhawk! (System {self.master.target_system}, Component {self.master.target_component})")
                
                self._request_data_streams()
                return True
        except Exception as e:
            logger.warning(f"Could not connect to Pixhawk hardware ({e}).")

        if self.sim_mode:
            logger.info("Initializing High-Fidelity Pixhawk MAVLink Hardware Simulator...")
            self.connected = True
            self.simulated = True
            self.state["mode"] = "SIM_STANDBY"
            self.state["lat"] = 37.774929
            self.state["lon"] = -122.419416
            self.state["alt"] = 0.0
            self.state["gps_fix"] = 3
            self.state["satellites_visible"] = 14
            return True
        return False

    def _request_data_streams(self):
        if not self.master:
            return
        try:
            from pymavlink import mavutil
            rate = self.config.get("gps_stream_rate_hz", 10)
            logger.info(f"Requesting MAVLink streams from Pixhawk at {rate}Hz...")
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                rate, 1
            )
        except Exception as e:
            logger.error(f"Error requesting data streams: {e}")

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self.thread.start()
        logger.info("Mavlink controller background telemetry loop started.")

    def _telemetry_loop(self):
        while self.running:
            now = time.time()
            if now - self.last_heartbeat_sent >= 1.0:
                self._send_companion_heartbeat()
                self.last_heartbeat_sent = now

            if self.simulated:
                self._update_simulated_telemetry()
            else:
                self._read_mavlink_messages()
            
            self.check_gcs_failsafe()
            time.sleep(0.05)

    def _send_companion_heartbeat(self):
        if self.simulated or not self.master:
            return
        try:
            from pymavlink import mavutil
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
        except Exception as e:
            logger.debug(f"Heartbeat send failed: {e}")

    def touch_gcs_heartbeat(self):
        with self.lock:
            self.last_gcs_ping = time.time()
            self.state["gcs_connected"] = True

    def check_gcs_failsafe(self):
        if time.time() - self.last_gcs_ping > self.gcs_timeout:
            with self.lock:
                if self.state["gcs_connected"]:
                    logger.warning(f"GCS TELEMETRY LINK TIMEOUT (>{self.gcs_timeout}s)! Engaging LOITER failsafe.")
                    self.state["gcs_connected"] = False
                    if self.state["armed"] and self.state["mode"] == MAV_MODE_GUIDED:
                        self.set_mode(MAV_MODE_LOITER)

    def _update_simulated_telemetry(self):
        with self.lock:
            if self.state["armed"] and self.state["mode"] == MAV_MODE_GUIDED:
                dt = 0.05
                deg_per_meter = 1.0 / 111000.0
                self.state["lat"] += self.state["vx"] * dt * deg_per_meter
                self.state["lon"] += self.state["vy"] * dt * deg_per_meter
                self.state["alt"] = max(0.0, self.state["alt"] + self.state["vz"] * dt)

    def _read_mavlink_messages(self):
        if not self.master:
            return
        try:
            msg = self.master.recv_match(blocking=False)
            if not msg:
                return

            msg_type = msg.get_type()
            with self.lock:
                if msg_type == "HEARTBEAT":
                    from pymavlink import mavutil
                    self.state["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    mode_str = mavutil.mode_string_v10(msg)
                    if self.state["mode"] == MAV_MODE_GUIDED and mode_str not in [MAV_MODE_GUIDED, "GUIDED"]:
                        logger.warning(f"MANUAL RC OVERRIDE DETECTED! Mode changed from GCS to {mode_str}")
                        self.state["manual_override"] = True
                    self.state["mode"] = mode_str

                elif msg_type == "GLOBAL_POSITION_INT":
                    self.state["lat"] = msg.lat / 1e7
                    self.state["lon"] = msg.lon / 1e7
                    self.state["alt"] = msg.relative_alt / 1000.0
                    self.state["heading"] = msg.hdg / 100.0

                elif msg_type == "GPS_RAW_INT":
                    self.state["gps_fix"] = getattr(msg, "fix_type", 0)
                    self.state["satellites_visible"] = getattr(msg, "satellites_visible", 0)

                elif msg_type == "RC_CHANNELS":
                    rc_ch5 = getattr(msg, "chan5_raw", 1500)
                    if rc_ch5 > 1700:
                        if not self.state["manual_override"]:
                            logger.warning("R12 Receiver RC Switch engaged MANUAL OVERRIDE.")
                            self.state["manual_override"] = True
                    else:
                        self.state["manual_override"] = False

                elif msg_type == "SYS_STATUS":
                    self.state["battery_pct"] = msg.battery_remaining
        except Exception as e:
            logger.error(f"Error reading MAVLink packet: {e}")

    def set_mode(self, mode_name):
        logger.info(f"Requesting flight mode: {mode_name}")
        with self.lock:
            self.state["mode"] = mode_name
            self.state["manual_override"] = False
            
        if self.simulated or not self.master:
            return True

        try:
            mode_id = self.master.mode_mapping().get(mode_name)
            if mode_id is None:
                logger.error(f"Unknown mode name {mode_name}")
                return False
            self.master.set_mode(mode_id)
            return True
        except Exception as e:
            logger.error(f"Set mode failed: {e}")
            return False

    def arm(self):
        logger.info("Arming Pixhawk motors...")
        with self.lock:
            self.state["armed"] = True
            
        if self.simulated or not self.master:
            return True

        try:
            self.master.arducopter_arm()
            self.master.motors_armed_wait()
            return True
        except Exception as e:
            logger.error(f"Arm command failed: {e}")
            return False

    def disarm(self):
        logger.info("Disarming Pixhawk motors...")
        with self.lock:
            self.state["armed"] = False
            
        if self.simulated or not self.master:
            return True

        try:
            self.master.arducopter_disarm()
            return True
        except Exception as e:
            logger.error(f"Disarm command failed: {e}")
            return False

    def takeoff(self, altitude=10.0):
        logger.info(f"Executing GUIDED Takeoff to {altitude}m...")
        self.set_mode(MAV_MODE_GUIDED)
        self.arm()

        with self.lock:
            self.state["vz"] = 1.0

        if self.simulated or not self.master:
            def simulate_climb():
                while self.state["alt"] < altitude and self.state["armed"]:
                    time.sleep(0.2)
                    self.state["alt"] = min(altitude, self.state["alt"] + 0.5)
                self.state["vz"] = 0.0
                logger.info(f"Target altitude {altitude}m reached (Simulated).")
            threading.Thread(target=simulate_climb, daemon=True).start()
            return True

        try:
            self.master.user_takeoff(altitude=altitude)
            return True
        except Exception as e:
            logger.error(f"Takeoff command failed: {e}")
            return False

    def send_velocity_target(self, target_vx, target_vy, target_vz, yaw_rate=0.0):
        if self.state["manual_override"]:
            logger.info("Velocity command suppressed: MANUAL OVERRIDE IS ACTIVE.")
            return

        dt = max(0.01, time.time() - self.last_pid_time)
        max_step = self.max_accel * dt

        with self.lock:
            curr_vx = self.state["vx"]
            curr_vy = self.state["vy"]
            curr_vz = self.state["vz"]

            smooth_vx = max(curr_vx - max_step, min(curr_vx + max_step, target_vx))
            smooth_vy = max(curr_vy - max_step, min(curr_vy + max_step, target_vy))
            smooth_vz = max(curr_vz - max_step, min(curr_vz + max_step, target_vz))

            self.state["vx"] = smooth_vx
            self.state["vy"] = smooth_vy
            self.state["vz"] = smooth_vz

        if self.simulated or not self.master:
            return

        try:
            from pymavlink import mavutil
            frame = (
                mavutil.mavlink.MAV_FRAME_BODY_NED 
                if self.mav_frame_str == "MAV_FRAME_BODY_NED" 
                else mavutil.mavlink.MAV_FRAME_LOCAL_NED
            )
            
            type_mask = (
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            )
            if yaw_rate != 0.0:
                type_mask &= ~mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE

            self.master.mav.set_position_target_local_ned_send(
                0,
                self.master.target_system,
                self.master.target_component,
                frame,
                type_mask,
                0, 0, 0,
                smooth_vx, smooth_vy, smooth_vz,
                0, 0, 0,
                0, yaw_rate
            )
        except Exception as e:
            logger.error(f"Error sending velocity vector: {e}")

    def track_target_offset(self, err_x, err_y):
        now = time.time()
        dt = max(0.01, now - self.last_pid_time)
        self.last_pid_time = now

        self.integral_x += err_x * dt
        self.integral_y += err_y * dt
        
        deriv_x = (err_x - self.prev_err_x) / dt
        deriv_y = (err_y - self.prev_err_y) / dt

        self.prev_err_x = err_x
        self.prev_err_y = err_y

        target_vx = -(self.pid_kp * err_y + self.pid_ki * self.integral_y + self.pid_kd * deriv_y) * 100.0
        target_vy = (self.pid_kp * err_x + self.pid_ki * self.integral_x + self.pid_kd * deriv_x) * 100.0

        yaw_rate = (self.pid_yaw_kp * err_x) * 10.0

        target_vx = max(-self.max_vel, min(self.max_vel, target_vx))
        target_vy = max(-self.max_vel, min(self.max_vel, target_vy))

        self.send_velocity_target(target_vx, target_vy, 0.0, yaw_rate=yaw_rate)
        self.estimate_target_gps(err_x, err_y)

    def estimate_target_gps(self, err_x, err_y):
        if self.state["lat"] == 0.0 or self.state["lon"] == 0.0:
            return

        alt = max(1.0, self.state["alt"])
        heading_rad = math.radians(self.state["heading"])

        offset_forward = -err_y * alt * 0.75
        offset_right = err_x * alt * 0.75

        dn = offset_forward * math.cos(heading_rad) - offset_right * math.sin(heading_rad)
        de = offset_forward * math.sin(heading_rad) + offset_right * math.cos(heading_rad)

        target_lat = self.state["lat"] + (dn / 111000.0)
        target_lon = self.state["lon"] + (de / (111000.0 * math.cos(math.radians(self.state["lat"]))))

        with self.lock:
            self.state["target_gps_lat"] = round(target_lat, 7)
            self.state["target_gps_lon"] = round(target_lon, 7)

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Mavlink controller stopped.")



