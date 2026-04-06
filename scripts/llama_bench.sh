#!/bin/bash
# llama_bench.sh — Run llama-bench on a fleet node and pull results back
#
# Produces standardised pp512/tg128 numbers comparable to published community
# benchmarks (llama.cpp, LLM-Perf Leaderboard, r/LocalLLaMA posts).
#
# Usage:
#   ./scripts/llama_bench.sh lenovo-gb10
#   ./scripts/llama_bench.sh lenovo-gb10 --models qwen3.5:27b,qwen3.5:9b
#   ./scripts/llama_bench.sh nvidia-ai --install     # also installs llama.cpp
#
# Prerequisites:
#   - SSH key in authorized_keys on target node
#   - GGUF models under ~/models/<model>/ (put there by setup script or HF download)
#   - Ollama does NOT need to be stopped — llama-bench uses GGUFs directly
#
# Output: results/llama_bench_<node>_<timestamp>.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_DIR="${ROOT}/config"
RESULTS_DIR="${ROOT}/results"

# ── Parse args ────────────────────────────────────────────────────────────────
NODE_NAME="${1:-}"
INSTALL=false
MODELS_OVERRIDE=""

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) INSTALL=true ;;
    --models) MODELS_OVERRIDE="$2"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
  shift
done

if [[ -z "$NODE_NAME" ]]; then
  echo "Usage: $0 <node-name> [--install] [--models model1,model2]"
  echo "Configured nodes:"
  python3 -c "
import yaml
with open('${CONFIG_DIR}/nodes.yaml') as f:
    cfg = yaml.safe_load(f)
for name, node in cfg['nodes'].items():
    print(f'  {name}: {node[\"host\"]} ({node.get(\"user\",\"?\")})')
"
  exit 1
fi

# ── Load node config ──────────────────────────────────────────────────────────
read -r HOST REMOTE_USER SSH_KEY OLLAMA_PORT NODE_TYPE <<< "$(python3 -c "
import yaml, sys
with open('${CONFIG_DIR}/nodes.yaml') as f:
    cfg = yaml.safe_load(f)
node = cfg['nodes'].get('${NODE_NAME}')
if not node:
    print('ERROR: unknown node ${NODE_NAME}', file=sys.stderr)
    sys.exit(1)
print(node['host'], node.get('user','ubuntu'), node.get('ssh_key','~/.ssh/id_ed25519'), node.get('ollama_port',11434), node.get('type','unknown'))
")"

SSH="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no ${REMOTE_USER}@${HOST}"
MODELS_DIR="/home/${REMOTE_USER}/models"
LLAMA_DIR="/home/${REMOTE_USER}/llama.cpp"

echo "=== llama-bench: ${NODE_NAME} (${REMOTE_USER}@${HOST}) ==="

# ── Install llama.cpp if needed ───────────────────────────────────────────────
if [[ "$INSTALL" == "true" ]]; then
  echo "--- Installing llama.cpp ---"
  $SSH "
    if [ -f ${LLAMA_DIR}/build/bin/llama-bench ]; then
      echo 'llama-bench already installed'
      exit 0
    fi

    # Dependencies
    sudo apt-get install -y cmake build-essential git libcurl4-openssl-dev 2>/dev/null

    # Clone and build
    git clone --depth 1 https://github.com/ggerganov/llama.cpp ${LLAMA_DIR} || \
      (cd ${LLAMA_DIR} && git pull)

    cd ${LLAMA_DIR}

    # Build with CUDA if available, otherwise CPU
    if command -v nvcc &>/dev/null || [ -d /usr/local/cuda ]; then
      echo 'Building with CUDA support...'
      cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    else
      echo 'Building CPU-only...'
      cmake -B build -DCMAKE_BUILD_TYPE=Release
    fi

    cmake --build build --config Release -j\$(nproc) --target llama-bench
    echo 'llama-bench built: '\$(${LLAMA_DIR}/build/bin/llama-bench --version 2>&1 | head -1)
  "
fi

# ── Verify llama-bench is available ──────────────────────────────────────────
$SSH "
  if [ ! -f ${LLAMA_DIR}/build/bin/llama-bench ]; then
    echo 'ERROR: llama-bench not found at ${LLAMA_DIR}/build/bin/llama-bench'
    echo 'Run with --install flag first'
    exit 1
  fi
  echo 'llama-bench: '\$(${LLAMA_DIR}/build/bin/llama-bench --version 2>&1 | head -1)
"

# ── Resolve model list ────────────────────────────────────────────────────────
if [[ -n "$MODELS_OVERRIDE" ]]; then
  # Manual override: map ollama-style names to GGUF paths
  MODELS_JSON=$(python3 -c "
import yaml, json
with open('${CONFIG_DIR}/models.yaml') as f:
    cfg = yaml.safe_load(f)

overrides = '${MODELS_OVERRIDE}'.split(',')
result = []
for tag in overrides:
    tag = tag.strip()
    slug = tag.replace(':', '-')
    # Find matching model config for HF filename
    gguf_file = None
    for tier in ['common', 'mid_tier', 'large']:
        for m in cfg.get('models', {}).get(tier, []):
            if m.get('ollama_tag','') == tag or m.get('name','') == tag:
                gguf_file = m.get('hf_file')
                break
    result.append({'tag': tag, 'slug': slug, 'gguf_file': gguf_file})
print(json.dumps(result))
")
else
  # Use all models assigned to this node that have GGUFs on disk
  MODELS_JSON=$(python3 -c "
import yaml, json
with open('${CONFIG_DIR}/models.yaml') as f:
    cfg = yaml.safe_load(f)

node_models = cfg.get('node_models', {}).get('${NODE_NAME}', [])
result = []
for tag in node_models:
    slug = tag.replace(':', '-')
    gguf_file = None
    for tier in ['common', 'mid_tier', 'large']:
        for m in cfg.get('models', {}).get(tier, []):
            if m.get('ollama_tag','') == tag or m.get('name','') == tag:
                gguf_file = m.get('hf_file')
                break
    result.append({'tag': tag, 'slug': slug, 'gguf_file': gguf_file})
print(json.dumps(result))
")
fi

echo "--- Models to benchmark ---"
echo "$MODELS_JSON" | python3 -c "import sys,json; [print(f'  {m[\"tag\"]}') for m in json.load(sys.stdin)]"

# ── Run llama-bench on each model ─────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_RESULTS="/tmp/llama_bench_${NODE_NAME}_${TIMESTAMP}.json"

$SSH "
  LLAMA_BENCH=${LLAMA_DIR}/build/bin/llama-bench
  MODELS_DIR=${MODELS_DIR}
  ALL_RESULTS='[]'

  run_bench() {
    local model_tag=\$1
    local gguf_path=\$2

    echo \"  Benchmarking: \${model_tag}\"
    echo \"  GGUF: \${gguf_path}\"

    if [ ! -f \"\${gguf_path}\" ]; then
      echo \"  SKIP — GGUF not found: \${gguf_path}\"
      return
    fi

    # Standard pp512/tg128 — matches llama.cpp community convention
    # -p 512: 512 prompt tokens (prefill)
    # -n 128: 128 generated tokens
    # -r 3:   3 repetitions, take median
    local raw
    raw=\$(\${LLAMA_BENCH} \\
      --model \"\${gguf_path}\" \\
      -p 512 -n 128 -r 3 \\
      --output json 2>/dev/null) || {
        echo \"  ERROR: llama-bench failed for \${model_tag}\"
        return
      }

    # Parse and annotate with metadata
    echo \"\${raw}\" | python3 -c \"
import sys, json
data = json.load(sys.stdin)
for entry in data:
    entry['node'] = '${NODE_NAME}'
    entry['model_tag'] = '\${model_tag}'
    entry['timestamp'] = '$(date -Iseconds)'
    entry['bench_tool'] = 'llama-bench'
    print(json.dumps(entry))
\"
  }

  echo '${MODELS_JSON}' | python3 -c \"
import sys, json, subprocess, os

models = json.load(sys.stdin)
results = []
models_dir = os.environ.get('MODELS_DIR', '${MODELS_DIR}')

for m in models:
    tag = m['tag']
    slug = tag.replace(':', '-')
    gguf_file = m.get('gguf_file')

    # Find the GGUF: check ~/models/<slug>/<file> or any .gguf in the dir
    model_dir = os.path.join(models_dir, slug)
    gguf_path = None

    if gguf_file:
        candidate = os.path.join(model_dir, gguf_file)
        if os.path.exists(candidate):
            gguf_path = candidate

    if not gguf_path and os.path.isdir(model_dir):
        ggufs = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
        if ggufs:
            gguf_path = os.path.join(model_dir, ggufs[0])

    if not gguf_path:
        print(f'  SKIP {tag} — no GGUF found in {model_dir}', flush=True)
        continue

    print(f'  Running llama-bench: {tag}', flush=True)
    result = subprocess.run(
        ['${LLAMA_DIR}/build/bin/llama-bench',
         '--model', gguf_path,
         '-p', '512', '-n', '128', '-r', '3',
         '--output', 'json'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr[:200]}', flush=True)
        continue

    try:
        data = json.loads(result.stdout)
        for entry in data:
            entry['node'] = '${NODE_NAME}'
            entry['model_tag'] = tag
            entry['timestamp'] = '$(date -Iseconds)'
            entry['bench_tool'] = 'llama-bench'
        results.extend(data)
        pp = next((e.get('avg_ts') for e in data if e.get('n_gen',0) == 0), None)
        tg = next((e.get('avg_ts') for e in data if e.get('n_gen',0) > 0), None)
        print(f'  {tag}: pp={pp} tok/s  tg={tg} tok/s', flush=True)
    except Exception as ex:
        print(f'  Parse error: {ex}', flush=True)

with open('${REMOTE_RESULTS}', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved: ${REMOTE_RESULTS}')
\"
"

# ── Pull results back ─────────────────────────────────────────────────────────
LOCAL_RESULTS="${RESULTS_DIR}/llama_bench_${NODE_NAME}_${TIMESTAMP}.json"
scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no \
  "${REMOTE_USER}@${HOST}:${REMOTE_RESULTS}" \
  "${LOCAL_RESULTS}" 2>/dev/null && \
  echo "Results pulled: ${LOCAL_RESULTS}" || \
  echo "WARNING: Could not pull results from ${HOST}:${REMOTE_RESULTS}"

echo ""
echo "=== llama-bench complete: ${NODE_NAME} ==="
echo "Results: ${LOCAL_RESULTS}"
echo "Generate report: python scripts/report_html.py --results results/"
