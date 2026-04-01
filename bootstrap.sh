#!/bin/bash
# bootstrap.sh — ai-bench node bootstrap
#
# Idempotent: safe to run on already-configured machines.
# Detects hardware automatically and configures accordingly.
#
# Usage (fresh machine):
#   git clone https://YOUR_PAT@github.com/wallysimmonds/ai-bench.git
#   cd ai-bench && ./bootstrap.sh
#
# Usage (test on existing machine):
#   ./bootstrap.sh --dry-run     # shows what it would do, changes nothing
#   ./bootstrap.sh --skip-models # skips model pulls (fast config-only run)

set -e

# ── Args ─────────────────────────────────────────────────────────────────────
DRY_RUN=false
SKIP_MODELS=false
for arg in "$@"; do
  case $arg in
    --dry-run)     DRY_RUN=true ;;
    --skip-models) SKIP_MODELS=true ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo "  $1"; }
ok()   { echo "  ✓ $1"; }
skip() { echo "  ~ $1 (already done)"; }
warn() { echo "  ⚠ $1"; }
die()  { echo "  ✗ $1"; exit 1; }
run()  { if $DRY_RUN; then echo "  [dry-run] $*"; else "$@"; fi; }

section() {
  echo ""
  echo "━━━ $1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── Hardware detection ────────────────────────────────────────────────────────
section "Hardware Detection"

ARCH=$(uname -m)
log "Architecture: $ARCH"

# GPU detection
HAS_NVIDIA=false
HAS_AMD=false
NODE_TYPE="cpu"

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
  if [ -n "$GPU_INFO" ]; then
    HAS_NVIDIA=true
    NODE_TYPE="nvidia"
    log "NVIDIA GPU(s) detected:"
    echo "$GPU_INFO" | while read line; do log "  $line"; done
  fi
fi

if [ "$HAS_NVIDIA" = false ] && command -v rocminfo &>/dev/null; then
  if rocminfo 2>/dev/null | grep -q "Device Type.*GPU"; then
    HAS_AMD=true
    NODE_TYPE="amd_unified"
    log "AMD GPU detected (ROCm)"
  fi
fi

# Unified memory detection (Strix Halo / GB10)
TOTAL_MEM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
log "System memory: ${TOTAL_MEM_GB}GB"

if [ "$TOTAL_MEM_GB" -ge 100 ]; then
  log "Large unified memory system detected"
  if [ "$ARCH" = "aarch64" ]; then
    NODE_TYPE="gb10"
    NODE_NAME="lenovo-gb10"
  elif [ "$HAS_AMD" = true ]; then
    NODE_TYPE="strix"
    NODE_NAME="bosgame-m5"
  fi
elif [ "$HAS_NVIDIA" = true ]; then
  NODE_TYPE="nvidia"
  NODE_NAME="nvidia-ai"
else
  NODE_NAME="unknown-node"
  warn "Could not identify node type — defaulting to CPU-only config"
fi

ok "Node type: $NODE_TYPE ($NODE_NAME)"

# ── Python dependencies ───────────────────────────────────────────────────────
section "Python Dependencies"

if python3 -c "import openpyxl, yaml, requests" &>/dev/null; then
  skip "Python packages already installed"
else
  log "Installing: openpyxl pyyaml requests"
  run pip install openpyxl pyyaml requests --break-system-packages -q
  ok "Python packages installed"
fi

# ── Ollama ────────────────────────────────────────────────────────────────────
section "Ollama"

if command -v ollama &>/dev/null; then
  OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
  skip "Ollama already installed ($OLLAMA_VER)"
else
  log "Installing Ollama..."
  if $DRY_RUN; then
    echo "  [dry-run] curl -fsSL https://ollama.com/install.sh | sh"
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  ok "Ollama installed"
fi

# Systemd override — idempotent (only writes if content differs)
OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="$OVERRIDE_DIR/override.conf"

build_override() {
  local models_path="$1"
  local extra="$2"
  cat << EOF
[Service]
Environment="OLLAMA_MODELS=$models_path"
Environment="OLLAMA_HOST=0.0.0.0:11434"
$extra
EOF
}

MODELS_PATH="$HOME/ollama-models"
run mkdir -p "$MODELS_PATH"

case "$NODE_TYPE" in
  strix|amd_unified)
    EXTRA='Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"'
    ;;
  *)
    EXTRA=""
    ;;
esac

DESIRED_OVERRIDE=$(build_override "$MODELS_PATH" "$EXTRA")
CURRENT_OVERRIDE=$(cat "$OVERRIDE_FILE" 2>/dev/null || echo "")

if [ "$DESIRED_OVERRIDE" = "$CURRENT_OVERRIDE" ]; then
  skip "Ollama systemd override unchanged"
else
  log "Writing Ollama systemd override..."
  run sudo mkdir -p "$OVERRIDE_DIR"
  if ! $DRY_RUN; then
    echo "$DESIRED_OVERRIDE" | sudo tee "$OVERRIDE_FILE" > /dev/null
  fi
  run sudo systemctl daemon-reload
  run sudo systemctl restart ollama
  sleep 3
  ok "Ollama service configured"
fi

# Ensure Ollama is running
if ! systemctl is-active --quiet ollama 2>/dev/null; then
  log "Starting Ollama service..."
  run sudo systemctl enable --now ollama
  sleep 3
fi
ok "Ollama service running"

# ── Node config ───────────────────────────────────────────────────────────────
section "Node Config"

if [ ! -f config/nodes.yaml ]; then
  if [ -f config/nodes.yaml.example ]; then
    run cp config/nodes.yaml.example config/nodes.yaml
    warn "Created config/nodes.yaml from example — update IPs before running benchmarks"
  fi
else
  skip "config/nodes.yaml already exists"
fi

# ── Model pulls ───────────────────────────────────────────────────────────────
section "Models"

if $SKIP_MODELS; then
  warn "Skipping model pulls (--skip-models)"
else
  # Common models — all nodes
  COMMON_MODELS=(
    "qwen2.5-coder:7b"
    "qwen2.5-coder:14b"
    "qwen3.5:9b"
  )

  # Mid-tier — needs 20GB+ VRAM or unified memory
  MID_MODELS=(
    "qwen2.5-coder:32b"
    "qwen3.5:35b-a3b"
  )

  # Large — unified memory nodes only
  LARGE_MODELS=(
    "qwen3.5:27b"
    "qwen3.5:72b"
    "qwen3-coder-next:80b-a3b-q4_K_M"
  )

  pull_if_missing() {
    local model="$1"
    if ollama list 2>/dev/null | grep -q "^${model}"; then
      skip "Model already pulled: $model"
    else
      log "Pulling: $model"
      run ollama pull "$model"
      ok "Pulled: $model"
    fi
  }

  log "Pulling common models (all nodes)..."
  for m in "${COMMON_MODELS[@]}"; do pull_if_missing "$m"; done

  # Smoke test before large pulls
  if ! $DRY_RUN; then
    log "Smoke testing qwen2.5-coder:7b..."
    SMOKE=$(ollama run qwen2.5-coder:7b "respond with only the word: ok" --nowordwrap 2>/dev/null | tr -d '[:space:]')
    if [ "$SMOKE" = "ok" ]; then
      ok "Smoke test passed"
    else
      warn "Smoke test response: '$SMOKE' (not exactly 'ok' — may still be working)"
    fi
  fi

  case "$NODE_TYPE" in
    nvidia)
      log "Pulling mid-tier models (NVIDIA ~40GB VRAM)..."
      for m in "${MID_MODELS[@]}"; do pull_if_missing "$m"; done
      ;;
    strix|gb10|amd_unified|unified_arm)
      log "Pulling mid-tier models..."
      for m in "${MID_MODELS[@]}"; do pull_if_missing "$m"; done
      log "Pulling large models (unified memory node)..."
      for m in "${LARGE_MODELS[@]}"; do pull_if_missing "$m"; done
      ;;
    *)
      warn "Unknown node type — skipping mid/large model pulls"
      ;;
  esac
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "Done"

echo ""
echo "  Node:      $NODE_NAME ($NODE_TYPE)"
echo "  Arch:      $ARCH"
echo "  Memory:    ${TOTAL_MEM_GB}GB"
echo "  Models:    $(ollama list 2>/dev/null | tail -n +2 | wc -l) installed"
echo ""

if [ -f config/nodes.yaml ]; then
  NEEDS_IP=$(grep -c "192.168.1.XX" config/nodes.yaml 2>/dev/null || echo 0)
  if [ "$NEEDS_IP" -gt 0 ]; then
    warn "config/nodes.yaml still has placeholder IPs — update before benchmarking"
  fi
fi

echo "  Next steps:"
echo "    Update config/nodes.yaml with node IPs"
echo "    python scripts/benchmark.py --node $NODE_NAME --suite standard"
echo "    python scripts/report_html.py --results results/"
echo ""

if $DRY_RUN; then
  warn "DRY RUN — no changes were made"
fi
