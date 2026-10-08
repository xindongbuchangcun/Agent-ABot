#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lifan/Benchmark}"
ABOT="${ROOT}/ABot-Navigation"
SCENES_ROOT="${ABOT_POI_SCENES_ROOT:-${ROOT}/data/ABotN-POIBench/occmaps}"
PYTHON_BIN="${ABOT_RENDER_PYTHON:-/home/lifan/miniconda3/bin/python}"

PORT="${PORT:-7036}"
GPUS="${GPUS:-0}"
MAX_SCENES_PER_GPU="${MAX_SCENES_PER_GPU:-1}"
RENDER_SCALE="1.5"

test -d "${SCENES_ROOT}" || {
  echo "POI scene directory not found: ${SCENES_ROOT}" >&2
  exit 1
}

cd "${ABOT}/render_server"
exec env PYTHON="${PYTHON_BIN}" bash run.sh \
  --port "${PORT}" \
  --gpus "${GPUS}" \
  --max_scenes_per_gpu "${MAX_SCENES_PER_GPU}" \
  --render_scale "${RENDER_SCALE}" \
  --scenes_root "${SCENES_ROOT}"
