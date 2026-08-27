#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -f "$SERVICE_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SERVICE_ROOT/.env"
  set +a
fi

MODEL_KEY="${1:-}"
case "$MODEL_KEY" in
  moss-realtime)
    PROFILE=moss
    BACKEND=moss
    MODEL_PATH_ENV=MOSS_MODEL_PATH
    PYTHON_BIN="${MOSS_PYTHON:-$SERVICE_ROOT/.venv-moss/bin/python}"
    MODEL_PATH="${MOSS_MODEL_PATH:-}"
    SERVED_MODEL_NAME="${MOSS_SERVED_MODEL_NAME:-OpenMOSS-Team/MOSS-VL-Realtime}"
    PORT="${MOSS_PORT:-8102}"
    GPU_SELECTION="${MOSS_GPUS:-${MOSS_GPU:-}}"
    DEVICE_MAP="${MOSS_DEVICE_MAP:-${MODEL_DEVICE_MAP:-balanced}}"
    NUM_FRAMES="${MOSS_NUM_FRAMES:-256}"
    NATIVE_FLAG=--native-realtime
    ;;
  mage)
    PROFILE=mage
    BACKEND=mage
    MODEL_PATH_ENV=MAGE_MODEL_PATH
    PYTHON_BIN="${MAGE_PYTHON:-$SERVICE_ROOT/.venv-mage/bin/python}"
    MODEL_PATH="${MAGE_MODEL_PATH:-}"
    SERVED_MODEL_NAME="${MAGE_SERVED_MODEL_NAME:-microsoft/Mage-VL}"
    PORT="${MAGE_PORT:-8103}"
    GPU_SELECTION="${MAGE_GPUS:-${MAGE_GPU:-}}"
    DEVICE_MAP="${MAGE_DEVICE_MAP:-${MODEL_DEVICE_MAP:-balanced}}"
    NUM_FRAMES="${MAGE_NUM_FRAMES:-32}"
    NATIVE_FLAG=--no-native-realtime
    ;;
  *)
    echo "Usage: bash scripts/run.sh {moss-realtime|mage}" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Run: bash scripts/setup_envs.sh $PROFILE" >&2
  exit 1
fi
if [[ -z "$MODEL_PATH" ]]; then
  echo "$MODEL_PATH_ENV must be set in moss_mage_api_service/.env" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Checkpoint directory not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ -n "$GPU_SELECTION" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_SELECTION"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PATH="$(dirname -- "$PYTHON_BIN"):${PATH}"
cd "$SERVICE_ROOT"
"$PYTHON_BIN" -m service.check_environment --profile "$PROFILE"
TORCH_LIB="$($PYTHON_BIN -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
echo "GPU visibility: ${CUDA_VISIBLE_DEVICES:-all physical GPUs}; device map: $DEVICE_MAP"

exec "$PYTHON_BIN" -m service.server \
  --backend "$BACKEND" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "${HOST:-127.0.0.1}" \
  --port "$PORT" \
  --device "${DEVICE:-cuda:0}" \
  --device-map "$DEVICE_MAP" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --fps "${FPS:-1.0}" \
  --num-frames "$NUM_FRAMES" \
  --max-pixels "${MAX_PIXELS:-150000}" \
  --mage-video-backend "${MAGE_VIDEO_BACKEND:-codec}" \
  --session-max-frames "${SESSION_MAX_FRAMES:-32}" \
  --response-wait-seconds "${RESPONSE_WAIT_SECONDS:-1.0}" \
  --max-tokens-per-second "${MAX_TOKENS_PER_SECOND:-12}" \
  --realtime-max-new-tokens "${REALTIME_MAX_NEW_TOKENS:-4096}" \
  --realtime-frame-queue-size "${REALTIME_FRAME_QUEUE_SIZE:-256}" \
  --mage-segment-seconds "${MAGE_SEGMENT_SECONDS:-8.0}" \
  --mage-segment-frames "${MAGE_SEGMENT_FRAMES:-16}" \
  --mage-cur-fps "${MAGE_CUR_FPS:-2.0}" \
  --mage-gate-threshold "${MAGE_GATE_THRESHOLD:-0.5}" \
  --mage-max-new-tokens "${MAGE_MAX_NEW_TOKENS:-80}" \
  --mage-max-segments "${MAGE_MAX_SEGMENTS:-0}" \
  --session-timeout-seconds "${SESSION_TIMEOUT_SECONDS:-900}" \
  "$NATIVE_FLAG"
