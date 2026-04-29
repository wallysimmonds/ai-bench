#!/bin/bash
# run_all.sh — Run full benchmark suite (Ollama + llama-bench) against a node
#
# Runs benchmark.py then llama_bench.sh for each model sequentially.
# Unloads each model from Ollama before llama_bench to avoid OOM on
# unified-memory nodes (GB10, Strix Halo).
#
# Usage:
#   ./scripts/run_all.sh lenovo-gb10
#   ./scripts/run_all.sh lenovo-gb10 --suite standard
#   ./scripts/run_all.sh lenovo-gb10 --models "qwen3.5:27b,llama3.3:70b"
#   ./scripts/run_all.sh lenovo-gb10 --skip-llama-bench
#
# GPU override flags (NVIDIA nodes only):
#   --cuda-devices 0              restrict Ollama to GPU 0 only
#   --cuda-devices 0,1            use both GPUs (default for multi-GPU nodes)
#   --force-spread                force tensor parallelism across all visible GPUs
#                                 (OLLAMA_SCHED_SPREAD=true)
#
# Examples:
#   # Single GPU baseline — models must fit in that GPU's VRAM
#   ./scripts/run_all.sh nvidia-ai --cuda-devices 0 --skip-llama-bench \
#     --models "qwen2.5-coder:7b,qwen2.5-coder:14b,qwen3.5:9b"
#
#   # Same models forced across both GPUs — measures tensor parallel overhead
#   ./scripts/run_all.sh nvidia-ai --cuda-devices 0,1 --force-spread \
#     --skip-llama-bench --models "qwen2.5-coder:7b,qwen2.5-coder:14b,qwen3.5:9b"
#
# Output: results/ (JSON) — then run report_html.py to generate HTML report
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${ROOT}/config"

# ── Parse args ────────────────────────────────────────────────────────────────
NODE_NAME="${1:-}"
SUITE="standard"
MODELS_OVERRIDE=""
SKIP_LLAMA_BENCH=false
CUDA_DEVICES=""
FORCE_SPREAD=false

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)             SUITE="$2"; shift ;;
    --models)            MODELS_OVERRIDE="$2"; shift ;;
    --skip-llama-bench)  SKIP_LLAMA_BENCH=true ;;
    --cuda-devices)      CUDA_DEVICES="$2"; shift ;;
    --force-spread)      FORCE_SPREAD=true ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
  shift
done

if [[ -z "$NODE_NAME" ]]; then
  echo "Usage: $0 <node-name> [--suite standard|coding] [--models m1,m2] [--skip-llama-bench]"
  echo "       [--cuda-devices <ids>] [--force-spread]"
  python3 -c "
import yaml
with open('${CONFIG_DIR}/nodes.yaml') as f:
    cfg = yaml.safe_load(f)
for name, node in cfg['nodes'].items():
    print(f'  {name}: {node[\"host\"]}')
"
  exit 1
fi

# ── Load node config ──────────────────────────────────────────────────────────
read -r HOST REMOTE_USER SSH_KEY OLLAMA_PORT <<< "$(python3 -c "
import yaml, sys
with open('${CONFIG_DIR}/nodes.yaml') as f:
    cfg = yaml.safe_load(f)
node = cfg['nodes'].get('${NODE_NAME}')
if not node:
    print('ERROR: unknown node', file=sys.stderr); sys.exit(1)
print(node['host'], node.get('user','ubuntu'), node.get('ssh_key','~/.ssh/id_ed25519'), node.get('ollama_port',11434))
")"

SSH="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no ${REMOTE_USER}@${HOST}"
OVERRIDE_FILE="/etc/systemd/system/ollama.service.d/override.conf"

# ── GPU override — temporarily patch Ollama systemd config ───────────────────
OLLAMA_OVERRIDE_PATCHED=false

patch_ollama_gpu() {
  local extra_env=""
  [[ -n "$CUDA_DEVICES" ]]   && extra_env+=$'\n'"Environment=\"CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}\""
  [[ "$FORCE_SPREAD" == "true" ]] && extra_env+=$'\n'"Environment=\"OLLAMA_SCHED_SPREAD=true\""
  [[ -z "$extra_env" ]] && return 0

  echo "  Patching Ollama GPU config on ${NODE_NAME}..."
  [[ -n "$CUDA_DEVICES" ]]        && echo "    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
  [[ "$FORCE_SPREAD" == "true" ]] && echo "    OLLAMA_SCHED_SPREAD=true"

  # Save original override so we can restore it
  ORIGINAL_OVERRIDE=$($SSH "cat ${OVERRIDE_FILE} 2>/dev/null || echo ''")

  $SSH "
    echo '${ORIGINAL_OVERRIDE}${extra_env}' | sudo tee ${OVERRIDE_FILE} > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
  "
  # Wait for Ollama to come back up
  local tries=0
  until curl -s "http://${HOST}:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1; do
    sleep 2; tries=$((tries+1))
    [[ $tries -ge 30 ]] && { echo "  ERROR: Ollama did not restart"; return 1; }
  done
  echo "  ✓ Ollama restarted with GPU override"
  OLLAMA_OVERRIDE_PATCHED=true
}

restore_ollama_gpu() {
  [[ "$OLLAMA_OVERRIDE_PATCHED" != "true" ]] && return 0
  echo "  Restoring Ollama GPU config on ${NODE_NAME}..."
  $SSH "
    echo '${ORIGINAL_OVERRIDE}' | sudo tee ${OVERRIDE_FILE} > /dev/null
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
  "
  echo "  ✓ Ollama config restored"
}

# Ensure we always restore the original config, even on failure
trap restore_ollama_gpu EXIT

patch_ollama_gpu

# ── Resolve model list ────────────────────────────────────────────────────────
if [[ -n "$MODELS_OVERRIDE" ]]; then
  IFS=',' read -ra MODELS <<< "$MODELS_OVERRIDE"
else
  readarray -t MODELS < <(python3 -c "
import yaml
with open('${CONFIG_DIR}/models.yaml') as f:
    cfg = yaml.safe_load(f)
for m in cfg.get('node_models', {}).get('${NODE_NAME}', []):
    print(m)
")
fi

# ── Header ────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  AI Fleet — Full Benchmark Run"
echo "  Node:   ${NODE_NAME} (${REMOTE_USER}@${HOST})"
echo "  Suite:  ${SUITE}"
echo "  Models: ${#MODELS[@]}"
[[ -n "$CUDA_DEVICES" ]]        && echo "  GPUs:   CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
[[ "$FORCE_SPREAD" == "true" ]] && echo "  Mode:   forced tensor parallel (SCHED_SPREAD)"
echo "========================================"
echo ""

PASS=0; FAIL=0; SKIPPED=0

for MODEL in "${MODELS[@]}"; do
  echo "----------------------------------------"
  echo "  Model: ${MODEL}"
  echo "----------------------------------------"

  # ── 1. Ollama benchmark ───────────────────────────────────────────────────
  echo "  [1/2] benchmark.py (Ollama)..."
  if python3 "${SCRIPT_DIR}/benchmark.py" --node "${NODE_NAME}" --model "${MODEL}" --suite "${SUITE}"; then
    echo "  ✓ benchmark.py done"
    ((PASS++)) || true
  else
    echo "  ✗ benchmark.py failed"
    ((FAIL++)) || true
  fi

  # ── Unload model from Ollama ──────────────────────────────────────────────
  OLLAMA_BASE="http://${HOST}:${OLLAMA_PORT}"
  echo "  Unloading ${MODEL} from Ollama..."
  curl -s -X POST "${OLLAMA_BASE}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"keep_alive\":0}" > /dev/null 2>&1 || true
  sleep 2

  # ── 2. llama-bench ────────────────────────────────────────────────────────
  if [[ "$SKIP_LLAMA_BENCH" == "true" ]]; then
    echo "  [2/2] llama_bench — skipped (--skip-llama-bench)"
    ((SKIPPED++)) || true
  else
    echo "  [2/2] llama_bench.sh..."
    if bash "${SCRIPT_DIR}/llama_bench.sh" "${NODE_NAME}" --models "${MODEL}" 2>&1; then
      echo "  ✓ llama_bench done"
      ((PASS++)) || true
    else
      echo "  ✗ llama_bench failed (GGUF incompatibility — Ollama vs llama.cpp format)"
      ((FAIL++)) || true
    fi
  fi

  echo ""
done

# ── Generate report ───────────────────────────────────────────────────────────
echo "========================================"
echo "  All models done."
echo "  Passed: ${PASS}  Failed: ${FAIL}  Skipped: ${SKIPPED}"
echo ""
echo "  Generating HTML report..."
python3 "${SCRIPT_DIR}/report_html.py" --results "${ROOT}/results/"
echo "========================================"
