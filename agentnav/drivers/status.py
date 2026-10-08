# agentnav/drivers/status.py
"""
robot_status — snapshot of robot state.
"""
from mcp.server.fastmcp import FastMCP

DRIVER_META = {
    "triggers": ["robot state", "where is the robot", "battery", "pose", "status", "how is the robot"],
    "safety_level": "safe",
    "phase": 1,
    "description": "Returns navigation state, pose, battery level, and velocity snapshot.",
}


def register(mcp: FastMCP, state, task_mgr, ros_client, meta=None) -> None:
    from agentnav.bridge_core.driver_meta import meta_suffix
    _sfx = meta_suffix(meta) if meta else ""

    def robot_status() -> dict:
        """
        Return the robot's current navigation state, odometry pose,
        battery level, and velocity.

        Response fields:
          nav_state       — idle | looking | planning | moving | arrived | failed
          pose            — {x, y, theta} in the robot's world frame (metres / radians)
          battery_pct     — battery percentage (0–100); -1 if /PowerVoltage not yet received
          battery_voltage — raw voltage from /PowerVoltage (V)
          velocity        — {v, w} linear (m/s) and angular (rad/s) velocity

        Call this before starting a navigation task to verify the robot is
        IDLE, or during a task alongside task_status for a fuller picture.
        """
        return {
            "nav_state": state.get_nav_state().value,
            "pose": state.pose,
            "battery_pct": state.battery_pct,
            "battery_voltage": state.battery_voltage,
            "velocity": state.velocity,
        }

    mcp.tool(description=(robot_status.__doc__ or "").strip() + _sfx)(robot_status)
