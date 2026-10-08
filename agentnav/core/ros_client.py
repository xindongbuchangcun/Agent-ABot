# agentnav/core/ros_client.py
"""
ROS2 interface — Phase 2 implementation.

Confirmed topic names (Wheeltec wheeltec_ros2, ROS2 Humble):

  Camera stack:
    /camera/color/image_raw       — RGB  (sensor_msgs/Image, rgb8 or bgr8)
    /camera/color/camera_info     — intrinsics K matrix (sensor_msgs/CameraInfo)
    /camera/depth/image_raw       — Depth (sensor_msgs/Image, 16UC1 mm or 32FC1 m)

  Chassis stack:
    /odom                         — wheel odometry (nav_msgs/Odometry)
    /cmd_vel                      — motion command (geometry_msgs/Twist)
    /PowerVoltage                 — battery voltage (std_msgs/Float32)

  TF:
    /tf, /tf_static               — camera_link → base_link (Phase 3)

Phase 3: pixel_to_pose() — add TF transform from camera_link to base_link.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agentnav.bridge_core.robot_state import RobotState

logger = logging.getLogger(__name__)

# ── Topic names (override via env vars if your robot differs) ─────────────────
TOPIC_COLOR_IMAGE = os.environ.get("TOPIC_COLOR_IMAGE", "/camera/color/image_raw")
TOPIC_COLOR_INFO  = os.environ.get("TOPIC_COLOR_INFO",  "/camera/color/camera_info")
TOPIC_DEPTH_IMAGE = os.environ.get("TOPIC_DEPTH_IMAGE", "/camera/depth/image_raw")
TOPIC_ODOM        = os.environ.get("TOPIC_ODOM",        "/odom")
TOPIC_CMD_VEL     = os.environ.get("TOPIC_CMD_VEL",     "/cmd_vel")
TOPIC_POWER       = os.environ.get("TOPIC_POWER",       "/PowerVoltage")

# ── Camera mounting offset (base_link frame) ──────────────────────────────────
# Forward distance from base_link origin to camera optical centre (metres).
# Set CAMERA_X_OFFSET if your camera is not directly above base_link.
CAMERA_X_OFFSET = float(os.environ.get("CAMERA_X_OFFSET", "0.0"))

# ── Battery voltage → percentage conversion ───────────────────────────────────
BATTERY_V_MIN = float(os.environ.get("BATTERY_V_MIN", "9.5"))   # 0 %
BATTERY_V_MAX = float(os.environ.get("BATTERY_V_MAX", "12.6"))  # 100 %

# ── rotate_to tuning ──────────────────────────────────────────────────────────
_ROTATE_KP        = 1.5              # proportional gain (rad/s per rad error)
_ROTATE_MAX_W     = 0.5              # max angular velocity (rad/s)
_ROTATE_TOLERANCE = math.radians(3)  # stop when within ±3°
_ROTATE_TIMEOUT   = 15.0             # seconds before giving up


def _angle_diff(target: float, current: float) -> float:
    """Shortest signed angular difference in (-π, π]."""
    d = target - current
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def _sample_depth_mm(depth, u: int, v: int, radius: int = 2) -> float:
    """
    Return depth at pixel (u, v) in millimetres using median of a neighbourhood.

    Samples a (2*radius+1)² patch centred on (u, v), keeps only pixels in
    [50 mm, 10 000 mm] (5 cm–10 m), and returns their median.

    Raises RuntimeError if the entire neighbourhood is invalid, so the agent
    gets a specific "no valid depth here" signal instead of a misleading
    "out of range" error from the downstream range check.
    """
    import numpy as np
    h, w = depth.shape
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float32)
    valid = patch[(patch >= 50) & (patch <= 10_000)]
    if valid.size == 0:
        raise RuntimeError(
            f"No valid depth within ±{radius} px of pixel ({u}, {v}). "
            "The region is likely a reflective/transparent surface (glass, mirror, "
            "polished floor) or beyond the depth sensor's 10 m range. "
            "Pick a pixel on a solid, matte, closer object — do not retry the same one."
        )
    return float(np.median(valid))


def _encode_jpeg(bgr_array) -> bytes:
    """Encode a BGR numpy uint8 array to JPEG bytes. Tries cv2, falls back to PIL."""
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            return bytes(buf)
    except ImportError:
        pass
    from PIL import Image as PILImage
    import io
    rgb = bgr_array[:, :, ::-1]  # BGR → RGB for PIL
    img = PILImage.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class RosClient:
    def __init__(self, state: "RobotState", camera_intrinsics: dict | None = None):
        """
        Args:
            state:             shared RobotState singleton
            camera_intrinsics: optional override dict {fx, fy, cx, cy}.
                               If absent, values are read from TOPIC_COLOR_INFO
                               automatically once the camera node is running.
                               CAMERA_FX/FY/CX/CY env vars also override.
        """
        self._state = state

        # Intrinsics: constructor > env var > auto from camera_info
        overrides = camera_intrinsics or {}
        def _get(key: str, env: str) -> Optional[float]:
            v = overrides.get(key, float(os.environ.get(env, "0")))
            return v if v > 0 else None

        self._fx = _get("fx", "CAMERA_FX")
        self._fy = _get("fy", "CAMERA_FY")
        self._cx = _get("cx", "CAMERA_CX")
        self._cy = _get("cy", "CAMERA_CY")
        self._intrinsics_ready = all(
            v is not None for v in (self._fx, self._fy, self._cx, self._cy)
        )

        self._node = None
        self._twist_pub = None
        self._spin_thread: Optional[threading.Thread] = None

        logger.info(
            "RosClient init — color=%s  depth=%s  intrinsics=%s",
            TOPIC_COLOR_IMAGE, TOPIC_DEPTH_IMAGE,
            "manual" if self._intrinsics_ready else "pending /camera/color/camera_info",
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Spin ROS2 subscriptions in a background daemon thread."""
        self._spin_thread = threading.Thread(
            target=self._spin, name="ros_client", daemon=True
        )
        self._spin_thread.start()
        logger.info("RosClient spin thread started")

    def stop(self) -> None:
        """Publish zero velocity and shut down the ROS2 node."""
        self.publish_stop()
        if self._node is not None:
            self._node.destroy_node()
            self._node = None

    def _spin(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image, CameraInfo
            from nav_msgs.msg import Odometry
            from std_msgs.msg import Float32
            from geometry_msgs.msg import Twist
        except ImportError as exc:
            logger.error(
                "rclpy unavailable: %s — "
                "source /opt/ros/humble/setup.bash before starting the bridge.", exc
            )
            return

        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node("agentnav_bridge")
        self._node = node
        self._twist_pub = node.create_publisher(Twist, TOPIC_CMD_VEL, 10)

        node.create_subscription(Image,      TOPIC_COLOR_IMAGE, self._on_color,       1)
        node.create_subscription(Image,      TOPIC_DEPTH_IMAGE, self._on_depth,       1)
        node.create_subscription(CameraInfo, TOPIC_COLOR_INFO,  self._on_camera_info, 1)
        node.create_subscription(Odometry,   TOPIC_ODOM,        self._on_odom,        1)
        node.create_subscription(Float32,    TOPIC_POWER,       self._on_power,       1)

        logger.info("ROS2 node 'agentnav_bridge' spinning")
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()

    # ── ROS2 callbacks ────────────────────────────────────────────────────────

    def _on_color(self, msg) -> None:
        try:
            import numpy as np
            enc = msg.encoding.lower()
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            if enc == "rgb8":
                arr = arr[:, :, ::-1].copy()  # RGB → BGR
            elif enc != "bgr8":
                logger.warning("Unexpected color encoding: %s", msg.encoding)
                return
            self._state.push_frame(_encode_jpeg(arr))
        except Exception:
            logger.exception("_on_color error")

    def _on_depth(self, msg) -> None:
        try:
            import numpy as np
            enc = msg.encoding.lower()
            if enc == "16uc1":
                arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(
                    msg.height, msg.width
                )
            elif enc == "32fc1":
                arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(
                    msg.height, msg.width
                )
                arr = (arr * 1000).astype(np.uint16)  # m → mm
            else:
                logger.warning("Unexpected depth encoding: %s", msg.encoding)
                return
            self._state.push_depth(arr)
        except Exception:
            logger.exception("_on_depth error")

    def _on_camera_info(self, msg) -> None:
        if self._intrinsics_ready:
            return
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1] row-major
        self._fx = msg.k[0]
        self._fy = msg.k[4]
        self._cx = msg.k[2]
        self._cy = msg.k[5]
        self._intrinsics_ready = True
        logger.info(
            "Camera intrinsics loaded from %s: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
            TOPIC_COLOR_INFO, self._fx, self._fy, self._cx, self._cy,
        )

    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._state.pose = {
            "x": round(p.x, 3),
            "y": round(p.y, 3),
            "theta": round(math.atan2(siny, cosy), 4),
        }
        t = msg.twist.twist
        self._state.velocity = {
            "v": round(t.linear.x, 3),
            "w": round(t.angular.z, 3),
        }

    def _on_power(self, msg) -> None:
        v = float(msg.data)
        self._state.battery_voltage = round(v, 2)
        pct = (v - BATTERY_V_MIN) / (BATTERY_V_MAX - BATTERY_V_MIN)
        self._state.battery_pct = max(0, min(100, int(pct * 100)))

    # ── Motion control ────────────────────────────────────────────────────────

    def publish_stop(self) -> None:
        """Publish a zero Twist to /cmd_vel immediately."""
        if self._twist_pub is None:
            logger.warning("publish_stop: ROS2 not connected yet")
            return
        from geometry_msgs.msg import Twist
        self._twist_pub.publish(Twist())

    def rotate_to(self, angle_deg: float) -> None:
        """
        Rotate to absolute odom-frame heading `angle_deg` degrees.
        Blocks until within ±3° of target or 15 s timeout.
        Used by robot_scan to step through scan angles.
        """
        if self._twist_pub is None:
            logger.warning("rotate_to: ROS2 not connected, skipping rotation")
            return

        from geometry_msgs.msg import Twist
        target = math.radians(angle_deg)
        t0 = time.monotonic()

        while time.monotonic() - t0 < _ROTATE_TIMEOUT:
            if self._state.should_stop:
                break
            err = _angle_diff(target, self._state.pose["theta"])
            if abs(err) < _ROTATE_TOLERANCE:
                break
            w = max(-_ROTATE_MAX_W, min(_ROTATE_MAX_W, _ROTATE_KP * err))
            cmd = Twist()
            cmd.angular.z = w
            self._twist_pub.publish(cmd)
            time.sleep(0.05)

        self._twist_pub.publish(Twist())  # stop rotation

    # ── Phase 3: pixel → pose ─────────────────────────────────────────────────

    def pixel_to_pose(self, u: int, v: int) -> dict:
        """
        Convert pixel (u, v) to robot base_link-frame pose {x, y, theta}.

        Camera optical frame → base_link frame mapping:
          x = z_cam + CAMERA_X_OFFSET   (depth is forward; add mounting offset)
          y = -x_cam                    (rightward in camera = leftward in base_link)
          theta = atan2(y, x)           (heading toward the target)

        For full TF support (non-horizontal camera, tilted mount), set
        CAMERA_X_OFFSET env var. Full /tf_static integration is a future item.

        Raises RuntimeError if depth is unavailable, intrinsics are not ready,
        or the depth value at (u, v) is out of valid range (0.05–10 m).
        """
        # Use the depth snapshot saved at robot_capture() time, not the current
        # latest_depth — the agent may have analysed an image from several frames ago.
        depth = self._state.captured_depth
        if depth is None:
            raise RuntimeError(
                "No captured depth frame. Call robot_capture() before pixel_to_pose() "
                f"to ensure depth is available (topic: {TOPIC_DEPTH_IMAGE})."
            )
        if not self._intrinsics_ready:
            raise RuntimeError(
                f"Camera intrinsics not ready. Waiting for {TOPIC_COLOR_INFO} "
                "or set CAMERA_FX/FY/CX/CY env vars."
            )

        d = _sample_depth_mm(depth, u, v) / 1000.0   # neighbourhood median, mm → m

        if d < 0.05 or d > 10.0:
            raise RuntimeError(
                f"Depth at pixel ({u}, {v}) is {d:.3f} m — out of valid range "
                "(0.05–10 m). The pixel may be on a reflective surface, too close, "
                "or missing depth data. Try a different pixel."
            )

        # Camera optical frame: z forward, x right, y down
        x_cam = (u - self._cx) * d / self._fx    # lateral (right positive)
        z_cam = d                                  # forward (depth)

        # base_link frame: x forward, y left
        x_base = z_cam + CAMERA_X_OFFSET
        y_base = -x_cam

        theta = math.atan2(y_base, x_base)
        return {"x": round(x_base, 3), "y": round(y_base, 3), "theta": round(theta, 4)}
