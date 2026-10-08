#!/usr/bin/env bash
# agentnav/scripts/start_robot_agent.sh
#
# Full setup + launch:
#   1. Check required env vars
#   2. Install agentnav into the navdp Python env (editable)
#   3. Generate ~/.nanobot/config.json from template (with env var substitution)
#   4. Run: nanobot gateway
#
# Prerequisites:
#   pip install nanobot-ai          (in the nanobot / Python 3.11 env)
#   source ~/.bashrc  (or export the env vars below)
#
# Required env vars:
#   ANTHROPIC_API_KEY   — your Anthropic API key
#   TELEGRAM_BOT_TOKEN  — from @BotFather
#   MY_TELEGRAM_ID      — your numeric Telegram user ID (from @userinfobot)
#   NAVDP_PYTHON        — path to Python 3.10 in the navdp conda/venv env
#                         e.g. /opt/conda/envs/navdp/bin/python
#
# Optional env vars (have defaults):
#   S1_MODE                 — nav2 | navdp  (default: nav2)
#   S1_CHECKPOINT           — path to NavDP checkpoint file
#   CAMERA_FX/FY/CX/CY     — camera intrinsics (default: 525/525/320/240)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_TEMPLATE="$REPO_ROOT/agentnav/config/nanobot_config.json"
NANOBOT_CONFIG="$HOME/.nanobot/config.json"

# ── 1. Check required env vars ─────────────────────────────────────────────────
echo "[setup] Checking required environment variables..."
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"
: "${TELEGRAM_BOT_TOKEN:?Set TELEGRAM_BOT_TOKEN}"
: "${MY_TELEGRAM_ID:?Set MY_TELEGRAM_ID (numeric, from @userinfobot)}"
: "${NAVDP_PYTHON:?Set NAVDP_PYTHON (path to python3.10 in navdp env)}"

# Defaults for optional vars
export S1_MODE="${S1_MODE:-nav2}"   # nav2 (default) | navdp
export S1_CHECKPOINT="${S1_CHECKPOINT:-}"
# Topic names default to Wheeltec layout — override if your robot differs:
# export TOPIC_COLOR_IMAGE=/camera/color/image_raw
# export TOPIC_DEPTH_IMAGE=/camera/depth/image_raw
# export TOPIC_ODOM=/odom  (confirm once nav stack is running)
# Camera intrinsics auto-read from /camera/color/camera_info — no need to set manually.

# ── 2. Install agentnav into the navdp Python env ──────────────────────────────
echo "[setup] Installing agentnav into navdp env: $NAVDP_PYTHON"
"$NAVDP_PYTHON" -m pip install -e "$REPO_ROOT/agentnav" --quiet

# ── 3. Generate ~/.nanobot/config.json ─────────────────────────────────────────
mkdir -p "$(dirname "$NANOBOT_CONFIG")"
echo "[setup] Writing $NANOBOT_CONFIG"

# Export so the Python subprocess can read them
export CONFIG_TEMPLATE NANOBOT_CONFIG

# Use Python to do env-var substitution (avoids envsubst dependency on Windows/macOS)
"$NAVDP_PYTHON" - <<'PYEOF'
import os, json, re, pathlib

template_path = os.environ["CONFIG_TEMPLATE"]
out_path = os.environ["NANOBOT_CONFIG"]

with open(template_path) as f:
    text = f.read()

# Replace ${VAR} with actual env var values
def replace(m):
    return os.environ.get(m.group(1), "")

text = re.sub(r'\$\{(\w+)\}', replace, text)
config = json.loads(text)

# Validate that critical fields are non-empty
assert config["providers"]["anthropic"]["apiKey"], "ANTHROPIC_API_KEY is empty"
assert config["channels"]["telegram"]["token"], "TELEGRAM_BOT_TOKEN is empty"
assert config["channels"]["telegram"]["allowFrom"][0], "MY_TELEGRAM_ID is empty"

with open(out_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"[setup] Config written to {out_path}")
PYEOF

# ── 4. Start nanobot gateway ───────────────────────────────────────────────────
echo "[setup] Starting nanobot gateway..."
echo "        Telegram: send a message to your bot to start chatting."
echo "        Available tools: robot_stop, robot_status, robot_capture, robot_scan, pixel_to_pose, s1_move, task_status, task_cancel"
echo ""
exec nanobot gateway --config "$NANOBOT_CONFIG"
