# agentnav/bridge_core/server.py
"""
agentnav MCP stdio server.

Runs in the navdp Python env (3.10).
Launched by nanobot as a subprocess — no manual management needed.

Hot-reload: dropping a new *.py file into agentnav/drivers/ and calling
_load_drivers() again will reload the driver source via importlib.reload.
Long-lived services (RosClient, S1Client) are singletons owned by this
module, so drivers never construct I/O in register().
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

# Prefer a proper `pip install -e agentnav/` install. Only fall back to a
# sys.path hack if the package is not importable, so dev clones still work.
try:
    import agentnav  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP

from agentnav.bridge_core.robot_state import RobotState
from agentnav.bridge_core.task_manager import TaskManager
from agentnav.core.ros_client import RosClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agentnav.server")

# ── Shared singletons ─────────────────────────────────────────────────────────
mcp = FastMCP("agentnav")

state = RobotState(
    s1_mode=os.environ.get("S1_MODE", "nav2"),
    s1_checkpoint=os.environ.get("S1_CHECKPOINT", ""),
)

task_mgr = TaskManager(state)

# Camera intrinsics: read from /camera/color/camera_info automatically (Phase 2).
# Set CAMERA_FX/FY/CX/CY env vars only if you need to override the live values.
ros_client = RosClient(state)
ros_client.start()  # subscribe camera + odom + power in background thread

# S1Client is a long-lived service (Nav2 action client + background rclpy spin).
# Owned here so driver hot-reload never constructs a second instance.
s1_client = None
if state.s1_mode == "nav2":
    from agentnav.core.s1_client import S1Client
    from agentnav.bridge_core.telegram_notifier import TelegramNotifier

    s1_client = S1Client(state, task_mgr, notifier=TelegramNotifier.from_env())
    s1_client.start()
else:
    logger.warning(
        "server: S1_MODE=%r is not yet implemented — s1_move will return an error.",
        state.s1_mode,
    )

# Exposed so drivers can pull service references without constructing anything.
# Drivers must treat this as read-only.
services: dict = {
    "s1_client": s1_client,
}

# ── Driver hot-loader ──────────────────────────────────────────────────────────
DRIVERS_DIR = Path(__file__).parent.parent / "drivers"


def _load_drivers() -> list[str]:
    """
    Import (or reload) every non-private *.py in drivers/ and call register().

    Safe to call multiple times. On subsequent calls, modules already present
    in sys.modules are refreshed via importlib.reload so new source takes
    effect. Register() is expected to be side-effect-free: it only wires MCP
    tools — it must NOT create I/O singletons (put those in server.py).
    """
    from agentnav.bridge_core.driver_meta import validate_driver_meta

    loaded: list[str] = []
    for f in sorted(DRIVERS_DIR.glob("*.py")):
        if f.name.startswith("_"):
            continue
        module_name = f"agentnav.drivers.{f.stem}"
        try:
            if module_name in sys.modules:
                mod = importlib.reload(sys.modules[module_name])
            else:
                mod = importlib.import_module(module_name)
            if hasattr(mod, "register"):
                meta = getattr(mod, "DRIVER_META", None)
                if meta is not None:
                    validate_driver_meta(meta, f.stem)
                mod.register(mcp, state, task_mgr, ros_client, meta)
                loaded.append(f.stem)
        except Exception:
            logger.exception("Failed to load driver: %s", f.name)
    logger.info("Loaded drivers: %s", loaded)
    return loaded


_load_drivers()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting agentnav MCP server (stdio)")
    mcp.run(transport="stdio")
