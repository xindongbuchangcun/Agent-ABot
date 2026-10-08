# agentnav/bridge_core/driver_meta.py
"""
Driver metadata schema and helpers.

Each driver module may export a module-level DRIVER_META dict:

    DRIVER_META = {
        "triggers":      ["stop", "halt", "emergency"],  # list[str], non-empty
        "safety_level":  "danger",                       # "safe" | "caution" | "danger"
        "phase":         1,                              # 1 | 2 | 3
        "description":   "One-line summary.",            # str
    }

server.py reads DRIVER_META during driver loading, validates it, and passes
it into register() so drivers can embed the metadata in MCP tool descriptions.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

REQUIRED_KEYS: frozenset[str] = frozenset({"triggers", "safety_level", "phase", "description"})
VALID_SAFETY_LEVELS: frozenset[str] = frozenset({"safe", "caution", "danger"})
VALID_PHASES: frozenset[int] = frozenset({1, 2, 3})


def validate_driver_meta(meta: dict, driver_name: str) -> bool:
    """
    Validate a DRIVER_META dict.  Logs warnings for each violation.
    Returns True if valid, False otherwise.  Never raises.
    """
    ok = True

    missing = REQUIRED_KEYS - meta.keys()
    if missing:
        logger.warning("Driver '%s' DRIVER_META missing keys: %s", driver_name, sorted(missing))
        ok = False

    if meta.get("safety_level") not in VALID_SAFETY_LEVELS:
        logger.warning(
            "Driver '%s' invalid safety_level '%s' (expected one of %s)",
            driver_name, meta.get("safety_level"), sorted(VALID_SAFETY_LEVELS),
        )
        ok = False

    if meta.get("phase") not in VALID_PHASES:
        logger.warning(
            "Driver '%s' invalid phase '%s' (expected one of %s)",
            driver_name, meta.get("phase"), sorted(VALID_PHASES),
        )
        ok = False

    triggers = meta.get("triggers", [])
    if not isinstance(triggers, list) or not triggers:
        logger.warning("Driver '%s' 'triggers' must be a non-empty list", driver_name)
        ok = False

    desc = meta.get("description", "")
    if not isinstance(desc, str) or not desc.strip():
        logger.warning("Driver '%s' 'description' must be a non-empty string", driver_name)
        ok = False

    return ok


def meta_suffix(meta: dict) -> str:
    """
    Return a one-line suffix to append to MCP tool descriptions.

    Example output:
        \\n[safety:danger | phase:1 | triggers: stop, halt, emergency]
    """
    triggers = ", ".join(meta.get("triggers", []))
    safety = meta.get("safety_level", "?")
    phase = meta.get("phase", "?")
    return f"\n[safety:{safety} | phase:{phase} | triggers: {triggers}]"
