# agentnav/drivers/ros_introspect.py
"""
ROS2 dynamic node discovery tools.

Wraps the `ros2` CLI via subprocess so the Agent can autonomously explore
any robot's nodes, topics, and services without hardcoded knowledge.

Requires: ros2 CLI available in PATH (source /opt/ros/$ROS_DISTRO/setup.bash).
Gracefully returns {"error": "..."} dicts when ROS2 is unavailable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10
_ROS_DISTRO = os.environ.get("ROS_DISTRO", "humble")
_ROS2_SETUP_HINT = f"source /opt/ros/{_ROS_DISTRO}/setup.bash before starting the bridge"

# Topics whose publish could cause unintended robot motion
_DANGEROUS_TOPICS: frozenset[str] = frozenset({
    "/cmd_vel",
    "/cmd_vel_nav",
    "/cmd_vel_teleop",
    "/cmd_vel_mux/input/navi",
    "/mobile_base/commands/velocity",
    "/robot/cmd_vel",
    "/diff_drive/cmd_vel",
    "/velocity_command",
})
_DANGEROUS_PATTERNS = (
    re.compile(r".*/cmd_vel.*"),
    re.compile(r".*velocity_command.*"),
    re.compile(r".*motor_cmd.*"),
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _check_ros2() -> Optional[str]:
    """Return error string if ros2 is not available, else None."""
    if shutil.which("ros2") is None:
        return (
            f"ROS2 not available: 'ros2' binary not found in PATH. "
            f"{_ROS2_SETUP_HINT}."
        )
    return None


def _run_ros2(*args: str, timeout: int = _DEFAULT_TIMEOUT) -> tuple[str, str, int]:
    """Run `ros2 <args>` synchronously. Returns (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            ["ros2", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "subprocess timeout", -1
    except FileNotFoundError:
        return "", "'ros2' not found", 127


async def _arun_ros2(*args: str, timeout: int = _DEFAULT_TIMEOUT) -> tuple[str, str, int]:
    """Async wrapper around _run_ros2 using asyncio.to_thread."""
    return await asyncio.to_thread(_run_ros2, *args, timeout=timeout)


def _to_yaml_str(data: dict) -> str:
    """Convert dict to YAML string for ros2 CLI args. Falls back to JSON."""
    try:
        import yaml  # PyYAML — present on all ROS2 installations
        return yaml.dump(data, default_flow_style=True).strip()
    except ImportError:
        return json.dumps(data)


def _warn_dangerous_topic(topic: str) -> Optional[str]:
    """Return a safety warning string if topic is motion-related, else None."""
    if topic in _DANGEROUS_TOPICS:
        return (
            f"'{topic}' is a motion-control topic. "
            "Verify robot_status() is IDLE and area is clear before publishing. "
            "For emergency stop, use robot_stop() instead."
        )
    for pattern in _DANGEROUS_PATTERNS:
        if pattern.match(topic):
            return (
                f"'{topic}' matches a motion-related topic pattern. "
                "Verify robot state before publishing velocity commands."
            )
    return None


def _domain_id() -> str:
    return os.environ.get("ROS_DOMAIN_ID", "0")


def _parse_topic_list(output: str, show_types: bool) -> list:
    topics = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if show_types:
            # Format: "/topic_name [msg/Type]"
            m = re.match(r"^(\S+)\s+\[([^\]]+)\]$", line)
            if m:
                topics.append({"name": m.group(1), "type": m.group(2)})
            else:
                topics.append({"name": line, "type": "unknown"})
        else:
            topics.append(line)
    return topics


def _parse_topic_info(output: str) -> dict:
    """Parse `ros2 topic info -v` output into a structured dict."""
    result: dict = {
        "type": "",
        "publisher_count": 0,
        "subscriber_count": 0,
        "publishers": [],
        "subscribers": [],
    }
    section = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Type:"):
            result["type"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Publisher count:"):
            result["publisher_count"] = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("Subscription count:"):
            result["subscriber_count"] = int(stripped.split(":", 1)[1].strip())
        elif "Publisher Details" in stripped or "Node name:" in stripped:
            if "Publisher Details" in stripped:
                section = "publishers"
            elif section == "publishers" and stripped.startswith("Node name:"):
                result["publishers"].append(stripped.split(":", 1)[1].strip())
        if "Subscription Details" in stripped:
            section = "subscribers"
        elif section == "subscribers" and stripped.startswith("Node name:"):
            result["subscribers"].append(stripped.split(":", 1)[1].strip())
    return result


# ── Driver registration ───────────────────────────────────────────────────────

DRIVER_META = {
    "triggers": ["ros nodes", "ros topics", "inspect robot", "what topics", "ros services",
                 "list nodes", "discover robot", "ros2"],
    "safety_level": "caution",
    "phase": 1,
    "description": "ROS2 dynamic discovery: list nodes, topics, services, echo/pub messages.",
}


def register(mcp: FastMCP, state, task_mgr, ros_client, meta=None) -> None:  # noqa: ARG001
    from agentnav.bridge_core.driver_meta import meta_suffix
    _sfx = meta_suffix(meta) if meta else ""

    async def ros_list_nodes() -> dict:
        """
        List all running ROS2 nodes on the current ROS_DOMAIN_ID.

        Returns:
          nodes           — list of node name strings
          count           — number of nodes found
          ros_domain_id   — active ROS_DOMAIN_ID (0 if not set)

        Call this first when connecting to an unknown robot to understand
        its topology. If the list is empty, check ros_domain_id matches
        the robot's domain.
        """
        err = _check_ros2()
        if err:
            return {"error": err, "ros_domain_id": _domain_id()}
        stdout, stderr, rc = await _arun_ros2("node", "list")
        if rc != 0:
            return {"error": "ros2 command failed", "stderr": stderr,
                    "returncode": rc, "ros_domain_id": _domain_id()}
        nodes = [l.strip() for l in stdout.splitlines() if l.strip()]
        return {"nodes": nodes, "count": len(nodes), "ros_domain_id": _domain_id()}

    mcp.tool(description=(ros_list_nodes.__doc__ or "").strip() + _sfx)(ros_list_nodes)

    @mcp.tool()
    async def ros_list_topics(show_types: bool = False) -> dict:
        """
        List all active ROS2 topics.

        Args:
          show_types — if True, include message type for each topic

        Returns (show_types=False):
          topics  — list of topic name strings, e.g. ['/cmd_vel', '/odom']

        Returns (show_types=True):
          topics  — list of {"name": "...", "type": "..."} dicts

        Use show_types=True after ros_list_nodes() to understand what data
        each topic carries before calling ros_topic_echo or ros_topic_pub.
        """
        err = _check_ros2()
        if err:
            return {"error": err, "ros_domain_id": _domain_id()}
        args = ["topic", "list", "-t"] if show_types else ["topic", "list"]
        stdout, stderr, rc = await _arun_ros2(*args)
        if rc != 0:
            return {"error": "ros2 command failed", "stderr": stderr,
                    "returncode": rc, "ros_domain_id": _domain_id()}
        topics = _parse_topic_list(stdout, show_types)
        return {"topics": topics, "count": len(topics), "ros_domain_id": _domain_id()}

    @mcp.tool()
    async def ros_topic_info(topic: str) -> dict:
        """
        Get detailed information about a ROS2 topic.

        Args:
          topic — full topic name with leading slash (e.g. '/cmd_vel')

        Returns:
          topic             — the topic name
          type              — message type string
          publisher_count   — number of nodes publishing
          subscriber_count  — number of nodes subscribing
          publishers        — list of publisher node names
          subscribers       — list of subscriber node names
          warning           — (optional) safety warning for motion-related topics

        Use this before ros_topic_pub or ros_topic_echo to understand
        a topic's data flow and message type.
        """
        err = _check_ros2()
        if err:
            return {"error": err, "ros_domain_id": _domain_id()}
        stdout, stderr, rc = await _arun_ros2("topic", "info", "-v", topic)
        if rc != 0:
            return {"error": "ros2 command failed", "stderr": stderr,
                    "returncode": rc, "ros_domain_id": _domain_id()}
        info = _parse_topic_info(stdout)
        result: dict = {"topic": topic, **info, "ros_domain_id": _domain_id()}
        warning = _warn_dangerous_topic(topic)
        if warning:
            result["warning"] = warning
        return result

    @mcp.tool()
    async def ros_topic_echo(topic: str, timeout_s: int = 10) -> dict:
        """
        Read one message from a ROS2 topic.

        Args:
          topic      — full topic name with leading slash
          timeout_s  — max seconds to wait for a message (default 10).
                       Use 15–30 for slow topics (e.g. /battery_state at 0.1Hz).

        Returns:
          topic    — the topic name
          message  — dict representation of the received message

        Returns {"error": "timeout"} if no message arrives within timeout_s.
        Call ros_topic_info first to confirm the topic has active publishers.
        """
        err = _check_ros2()
        if err:
            return {"error": err, "ros_domain_id": _domain_id()}
        stdout, stderr, rc = await _arun_ros2(
            "topic", "echo", "--once", topic,
            timeout=timeout_s + 2,  # slight buffer over subprocess timeout
        )
        if rc == -1 and "timeout" in stderr:
            return {
                "error": "timeout",
                "timeout_s": timeout_s,
                "hint": "No message received. Check ros_topic_info to confirm publishers exist.",
                "ros_domain_id": _domain_id(),
            }
        if rc != 0:
            return {"error": "ros2 command failed", "stderr": stderr,
                    "returncode": rc, "ros_domain_id": _domain_id()}
        # `ros2 topic echo --once` outputs YAML; parse it
        try:
            import yaml
            message = yaml.safe_load(stdout) or {}
        except Exception:
            message = {"raw": stdout.strip()}
        return {"topic": topic, "message": message, "ros_domain_id": _domain_id()}

    @mcp.tool()
    async def ros_service_list() -> dict:
        """
        List all available ROS2 services with their types.

        Returns:
          services  — list of {"name": "...", "type": "..."} dicts
          count     — number of services

        Use ros_service_call() to invoke a service once you know its name
        and type.
        """
        err = _check_ros2()
        if err:
            return {"error": err, "ros_domain_id": _domain_id()}
        stdout, stderr, rc = await _arun_ros2("service", "list", "-t")
        if rc != 0:
            return {"error": "ros2 command failed", "stderr": stderr,
                    "returncode": rc, "ros_domain_id": _domain_id()}
        services = _parse_topic_list(stdout, show_types=True)
        return {"services": services, "count": len(services), "ros_domain_id": _domain_id()}

    @mcp.tool()
    async def ros_topic_pub(
        topic: str,
        msg_type: str,
        data: dict,
        once: bool = True,
    ) -> dict:
        """
        Publish a message to a ROS2 topic.

        Args:
          topic     — full topic name (e.g. '/cmd_vel')
          msg_type  — ROS2 message type (e.g. 'geometry_msgs/msg/Twist')
          data      — message fields as a dict matching the message schema
          once      — if True (default), publish once and exit

        Returns:
          published  — True on success
          topic      — the topic name
          warning    — always present for motion-related topics

        SAFETY: Motion-related topics (e.g. /cmd_vel) always return a
        warning. For emergency stop, use robot_stop() not this tool.

        Example:
          ros_topic_pub('/cmd_vel', 'geometry_msgs/msg/Twist',
                        {'linear': {'x': 0.0}, 'angular': {'z': 0.0}})
        """
        err = _check_ros2()
        if err:
            return {"error": err, "ros_domain_id": _domain_id()}
        yaml_data = _to_yaml_str(data)
        args = ["topic", "pub"]
        if once:
            args.append("--once")
        args += [topic, msg_type, yaml_data]
        stdout, stderr, rc = await _arun_ros2(*args)
        if rc != 0:
            return {"error": "ros2 command failed", "stderr": stderr,
                    "returncode": rc, "ros_domain_id": _domain_id()}
        result: dict = {"published": True, "topic": topic, "ros_domain_id": _domain_id()}
        warning = _warn_dangerous_topic(topic)
        if warning:
            result["warning"] = warning
        return result

    @mcp.tool()
    async def ros_service_call(
        service: str,
        srv_type: str,
        args: Optional[dict] = None,
        timeout_s: int = _DEFAULT_TIMEOUT,
    ) -> dict:
        """
        Call a ROS2 service and return the response.

        Args:
          service    — full service name (e.g. '/clear_costmaps')
          srv_type   — service type (e.g. 'std_srvs/srv/Empty')
          args       — request fields as dict; use {} or None for empty requests
          timeout_s  — max seconds to wait for response (default 10)

        Returns:
          success   — True if service responded
          response  — dict representation of the service response
          service   — the service name

        Common Nav2 services:
          /clear_costmaps                    std_srvs/srv/Empty  {}
          /reinitialize_global_localization  std_srvs/srv/Empty  {}
        """
        err = _check_ros2()
        if err:
            return {"error": err, "ros_domain_id": _domain_id()}
        yaml_args = _to_yaml_str(args or {})
        stdout, stderr, rc = await _arun_ros2(
            "service", "call", service, srv_type, yaml_args,
            timeout=timeout_s + 2,
        )
        if rc == -1 and "timeout" in stderr:
            return {
                "error": "timeout",
                "timeout_s": timeout_s,
                "service": service,
                "hint": "Service may not exist or is not responding.",
                "ros_domain_id": _domain_id(),
            }
        if rc != 0:
            return {"error": "ros2 command failed", "stderr": stderr,
                    "returncode": rc, "ros_domain_id": _domain_id()}
        try:
            import yaml
            response = yaml.safe_load(stdout) or {}
        except Exception:
            response = {"raw": stdout.strip()}
        return {"success": True, "response": response, "service": service,
                "ros_domain_id": _domain_id()}
