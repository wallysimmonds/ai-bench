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

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)          SUITE="$2"; shift ;;
    --models)         MODELS_OVERRIDE="$2"; shift ;;
    --skip-llama-bench) SKIP_LLAMA_BENCH=true ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
  shift
done

if [[ -z "$NODE_NAME" ]]; then
  echo "Usage: $0 <node-name> [--suite standard|coding] [--models m1,m2] [--skip-llama-bench]"
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

echo "========================================"
echo "  AI Fleet — Full Benchmark Run"
echo "  Node:   ${NODE_NAME} (${REMOTE_USER}@${HOST})"
echo "  Suite:  ${SUITE}"
echo "  Models: ${#MODELS[@]}"
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
