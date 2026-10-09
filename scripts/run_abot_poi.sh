#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTNAV="${AGENTNAV:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ROOT="${ROOT:-$(cd "${AGENTNAV}/.." && pwd)}"
ABOT="${ROOT}/ABot-Navigation"
DATA="${ABOT_ANNOTATION_DIR:-${ROOT}/data/ABotN-POIBench/annotations}"
MAPS="${ABOT_MAP_DIR:-${ROOT}/data/ABotN-POIBench/occmaps}"
OUTPUT="${OUTPUT:-${AGENTNAV}/outputs/poi_agentnav_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${ABOT_PYTHON:-/home/lifan/miniconda3/envs/abotn_eval/bin/python}"
DEPTH_GPU_ID="${ABOT_DEPTH_GPU_ID:-2}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:8000/v1/models}"
EXPECTED_VLLM_MODEL="${VLLM_SERVED_MODEL_NAME:-qwen3-vl-4b-instruct}"

# Keep every local service request off environment HTTP proxies.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

# Fail before creating task results if the configured model service is down
# or serves a different model name. Use stdlib only; curl is not guaranteed.
"${PYTHON_BIN}" - "${VLLM_HEALTH_URL}" "${EXPECTED_VLLM_MODEL}" <<'PY'
import json
import sys
from urllib.request import urlopen

url, expected = sys.argv[1:3]
with urlopen(url, timeout=10) as response:
    payload = json.load(response)
model_ids = [str(item.get("id", "")) for item in payload.get("data", [])]
if expected not in model_ids:
    raise SystemExit(
        f"vLLM model mismatch: expected {expected!r}, available={model_ids!r}"
    )
print(f"vLLM preflight OK: {expected}")
PY

# Evaluate an immutable per-run copy. Editor saves or other workspace changes
# during a long run cannot change the code/config represented by its metrics.
SNAPSHOT_ROOT="${OUTPUT}/_source_snapshot"
if [[ -e "${SNAPSHOT_ROOT}" ]]; then
  echo "Source snapshot already exists: ${SNAPSHOT_ROOT}" >&2
  exit 2
fi
mkdir -p "${SNAPSHOT_ROOT}"
cp -a "${AGENTNAV}/agentnav" "${SNAPSHOT_ROOT}/agentnav"
cp -a "${AGENTNAV}/nanobot" "${SNAPSHOT_ROOT}/nanobot"
SNAPSHOT_CONFIG="${SNAPSHOT_ROOT}/agentnav/abot/config/poi_agent.yaml"
sed -i "s|^workspace:.*|workspace: ${SNAPSHOT_ROOT}/agentnav/abot/workspace|" "${SNAPSHOT_CONFIG}"
find "${SNAPSHOT_ROOT}/agentnav" "${SNAPSHOT_ROOT}/nanobot" \
  -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.md' \) -print0 \
  | sort -z | xargs -0 sha256sum > "${SNAPSHOT_ROOT}/SHA256SUMS"

export PYTHONPATH="${SNAPSHOT_ROOT}:${ABOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${DEPTH_GPU_ID}"
cd "${ABOT}"
exec "${PYTHON_BIN}" -m abotn_evaluator.poi_goal.runner \
  --agent-module agentnav.abot.poi_agent:AgentNavPoiGoalAgent \
  --agent-config "${SNAPSHOT_CONFIG}" \
  --evaluator-module agentnav.abot.evaluator:AgentNavPoiGoalEvaluator \
  --data-dir "${DATA}" \
  --map-dir "${MAPS}" \
  --output-dir "${OUTPUT}" \
  --render-url "${RENDER_URL:-http://127.0.0.1:7036/render_gs}" \
  --max-steps "${MAX_STEPS:-100}" \
  "$@"
