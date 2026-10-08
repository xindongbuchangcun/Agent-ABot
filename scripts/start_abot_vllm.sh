#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/lifan/Benchmark/models/Qwen3-VL-4B-Instruct}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen3-vl-4b-instruct}"
PORT="${VLLM_PORT:-8000}"
GPU_IDS="${VLLM_GPU_IDS:-1}"
TP_SIZE="${VLLM_TP_SIZE:-1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-qwen3_xml}"
PYTHON_BIN="${VLLM_PYTHON:-/home/lifan/Benchmark/envs/qwen3_vl/bin/python}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONNOUSERSITE=1
exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype half \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image": 1}' \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --enforce-eager
