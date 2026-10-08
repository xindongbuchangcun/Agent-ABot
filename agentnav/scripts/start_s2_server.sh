#!/usr/bin/env bash
# agentnav/scripts/start_s2_server.sh
# Start the S2 vision-language model server.
# Requires: S2_MODEL (default: qwen3-vl), S2_PORT (default: 8890)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

S2_MODEL="${S2_MODEL:-qwen3-vl}"
S2_PORT="${S2_PORT:-8890}"

echo "[s2] Starting S2 server: model=$S2_MODEL port=$S2_PORT"
exec python "$REPO_ROOT/agentnav/server/s2_server.py" \
  --model "$S2_MODEL" \
  --port "$S2_PORT"
