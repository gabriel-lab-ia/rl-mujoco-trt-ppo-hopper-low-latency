#!/usr/bin/env bash
set -euo pipefail

ONNX_FILE="${1:-ppo_hopper.onnx}"
ENGINE_FILE="${2:-ppo_hopper_fp16.engine}"
LOG_FILE="${3:-trtexec_ppo_hopper_fp16.log}"

TRTEXEC_CANDIDATE="${TRTEXEC_BIN:-/usr/bin/trtexec}"
if [[ -x "${TRTEXEC_CANDIDATE}" ]]; then
  TRTEXEC="${TRTEXEC_CANDIDATE}"
elif command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="$(command -v trtexec)"
else
  echo "ERROR: trtexec not found. Checked ${TRTEXEC_CANDIDATE} and PATH."
  echo "Hint: export TRTEXEC_BIN=/full/path/to/trtexec"
  exit 1
fi

if [[ ! -f "${ONNX_FILE}" ]]; then
  echo "ERROR: ONNX file not found: ${ONNX_FILE}"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is required to inspect ONNX input metadata"
  exit 1
fi

INPUT_NAME="$(python - "$ONNX_FILE" <<'PY'
import sys
import onnx

m = onnx.load(sys.argv[1])
if not m.graph.input:
    raise SystemExit("No ONNX input found")
print(m.graph.input[0].name)
PY
)"

if [[ -z "${INPUT_NAME}" ]]; then
  echo "ERROR: Failed to resolve ONNX input name"
  exit 1
fi

echo "Using trtexec: ${TRTEXEC}"
echo "Building FP16 engine with input=${INPUT_NAME}, shape=1x11"
"${TRTEXEC}" \
  --onnx="${ONNX_FILE}" \
  --saveEngine="${ENGINE_FILE}" \
  --fp16 \
  --minShapes="${INPUT_NAME}:1x11" \
  --optShapes="${INPUT_NAME}:1x11" \
  --maxShapes="${INPUT_NAME}:1x11" \
  --builderOptimizationLevel=5 \
  --useCudaGraph \
  --noDataTransfers \
  --separateProfileRun \
  --verbose \
  | tee "${LOG_FILE}"

echo "Engine saved: ${ENGINE_FILE}"
echo "Log saved: ${LOG_FILE}"
