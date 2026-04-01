#!/bin/bash
# bootstrap.sh — ai-bench node bootstrap
#
# Idempotent: safe to run on already-configured machines.
# Detects hardware automatically and configures accordingly.
#
# Usage (fresh machine):
#   git clone https://github.com/wallysimmonds/ai-bench.git
#   cd ai-bench && ./bootstrap.sh
#
# Flags:
#   --dry-run          show what would happen, change nothing
#   --skip-models      skip model pulls (fast config-only run)
#   --with-webui       deploy Open WebUI in Docker after Ollama setup
#   --teardown-docker  stop/remove existing Docker Ollama container first

set -e

# ── Args ─────────────────────────────────────────────────────────────────────
DRY_RUN=false
SKIP_MODELS=false
WITH_WEBUI=false
TEARDOWN_DOCKER=false

for arg in "$@"; do
  case $arg in
    --dry-run)         DRY_RUN=true ;;
    --skip-models)     SKIP_MODELS=true ;;
    --with-webui)      WITH_WEBUI=true ;;
    --teardown-docker) TEARDOWN_DOCKER=true ;;
  esac
done

# ── tmux guard ───────────────────────────────────────────────────────────────
# Relaunch inside tmux so SSH disconnects don't kill long model pulls
if [ "$SKIP_MODELS" = false ] && [ "$DRY_RUN" = false ] && [ -z "$TMUX" ]; then
  if ! command -v tmux &>/dev/null; then
    echo "  Installing tmux..."
    sudo apt-get install -y tmux -q
  fi
  echo ""
  echo "  Launching inside tmux session 'bootstrap' to survive SSH disconnects."
  echo "  To reattach if disconnected: tmux attach -t bootstrap"
  echo ""
  sleep 1
  tmux new-session -s bootstrap "bash $0 $*; echo '--- bootstrap complete, press enter to close ---'; read"
  exit 0
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
log()  { echo "  $1"; }
ok()   { echo "  ✓ $1"; }
skip() { echo "  ~ $1 (already done)"; }
warn() { echo "  ⚠ $1"; }
run()  { if $DRY_RUN; then echo "  [dry-run] $*"; else "$@"; fi; }

section() {
  echo ""
  echo "━━━ $1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── Hardware detection ────────────────────────────────────────────────────────
section "Hardware Detection"

ARCH=$(uname -m)
log "Architecture: $ARCH"

HAS_NVIDIA=false
HAS_AMD=false
NODE_TYPE="cpu"
NODE_NAME="unknown-node"

if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
  if [ -n "$GPU_INFO" ]; then
    HAS_NVIDIA=true
    log "NVIDIA GPU(s) detected:"
    echo "$GPU_INFO" | while read line; do log "  $line"; done
  fi
fi

if [ "$HAS_NVIDIA" = false ] && command -v rocminfo &>/dev/null; then
  if rocminfo 2>/dev/null | grep -q "Device Type.*GPU"; then
    HAS_AMD=true
    log "AMD GPU detected (ROCm)"
  fi
fi

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
  else
    NODE_TYPE="unified"
    NODE_NAME="unified-node"
  fi
elif [ "$HAS_NVIDIA" = true ]; then
  NODE_TYPE="nvidia"
  NODE_NAME="nvidia-ai"
else
  warn "Could not identify node type — defaulting to CPU-only config"
fi

ok "Node type: $NODE_TYPE ($NODE_NAME)"

# ── Docker teardown (optional) ────────────────────────────────────────────────
section "Docker Check"

if command -v docker &>/dev/null; then
  # Find containers using port 11434
  DOCKER_OLLAMA=$(sudo docker ps --format "{{.Names}}\t{{.Ports}}" 2>/dev/null | \
    grep "11434" | awk '{print $1}' | head -1 || echo "")

  if [ -n "$DOCKER_OLLAMA" ]; then
    warn "Docker container using port 11434: $DOCKER_OLLAMA"
    if $TEARDOWN_DOCKER; then
      log "Stopping and removing: $DOCKER_OLLAMA"
      run sudo docker stop "$DOCKER_OLLAMA"
      run sudo docker rm "$DOCKER_OLLAMA"
      # Remove associated volume if it exists
      if sudo docker volume ls -q 2>/dev/null | grep -q "^${DOCKER_OLLAMA}$"; then
        run sudo docker volume rm "$DOCKER_OLLAMA"
        ok "Volume removed"
      fi
      ok "Docker container removed — port 11434 now free"
    else
      warn "Pass --teardown-docker to remove it (port conflict will prevent native Ollama)"
    fi
  else
    skip "No Docker container on port 11434"
  fi
else
  skip "Docker not installed"
fi

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

# Run install script every time — it handles upgrades idempotently
# Important for gpt-oss:120b which needs latest Ollama for MXFP4 support
if command -v ollama &>/dev/null; then
  log "Ollama $(ollama --version 2>/dev/null) — checking for updates..."
else
  log "Installing Ollama..."
fi

if $DRY_RUN; then
  echo "  [dry-run] curl -fsSL https://ollama.com/install.sh | sh"
else
  curl -fsSL https://ollama.com/install.sh | sh
fi
ok "Ollama up to date"

# Systemd override — idempotent, only writes if changed
OVERRIDE_DIR="/etc/systemd/system/ollama.service.d"
OVERRIDE_FILE="$OVERRIDE_DIR/override.conf"
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

DESIRED_OVERRIDE="[Service]
Environment=\"OLLAMA_MODELS=$MODELS_PATH\"
Environment=\"OLLAMA_HOST=0.0.0.0:11434\"
$EXTRA"

CURRENT_OVERRIDE=$(cat "$OVERRIDE_FILE" 2>/dev/null || echo "")

if [ "$DESIRED_OVERRIDE" = "$CURRENT_OVERRIDE" ]; then
  skip "Ollama systemd override unchanged"
else
  log "Writing Ollama systemd override..."
  run sudo mkdir -p "$OVERRIDE_DIR"
  if ! $DRY_RUN; then
    printf '%s\n' "$DESIRED_OVERRIDE" | sudo tee "$OVERRIDE_FILE" > /dev/null
  fi
  run sudo systemctl daemon-reload
  run sudo systemctl restart ollama
  sleep 3
  ok "Ollama service configured"
fi

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
    warn "Created config/nodes.yaml from example — update IPs before benchmarking"
  fi
else
  skip "config/nodes.yaml already exists"
fi

# ── Model pulls ───────────────────────────────────────────────────────────────
section "Models"

if $SKIP_MODELS; then
  warn "Skipping model pulls (--skip-models)"
else

  # Models are pulled to disk and benchmarked individually — not loaded concurrently

  # Common — all nodes
  COMMON_MODELS=(
    "qwen2.5-coder:7b"      # 5GB  — smoke test / fast iteration
    "qwen2.5-coder:14b"     # 9GB  — mid small
    "qwen3.5:9b"            # 6GB  — latest gen small
  )

  # Mid-tier — nvidia node (~40GB VRAM)
  MID_MODELS=(
    "qwen2.5-coder:32b"     # 20GB — strong coder
    "qwen3.5:35b-a3b"       # 24GB — MoE, 3B active, very efficient
  )

  # Large — unified memory nodes only (128GB)
  LARGE_MODELS=(
    "qwen3.5:27b"           # 17GB — general purpose baseline
    "qwen3.5:35b-a3b"       # 24GB — MoE efficiency reference
    "qwen3-coder-next"      # 48GB — key agentic coding benchmark (3B active)
    "gpt-oss:120b"          # 80GB — OpenAI open weight, MXFP4, needs latest Ollama
    "qwen3.5:122b"          # 81GB — flagship, tight on 128GB, pull last
  )

  pull_if_missing() {
    local model="$1"
    local model_base="${model%%:*}"
    if ollama list 2>/dev/null | grep -q "$model_base"; then
      skip "Already pulled: $model"
    else
      log "Pulling: $model"
      run ollama pull "$model"
      ok "Pulled: $model"
    fi
  }

  log "Pulling common models..."
  for m in "${COMMON_MODELS[@]}"; do pull_if_missing "$m"; done

  # Smoke test before committing to large downloads
  if ! $DRY_RUN; then
    log "Smoke testing qwen2.5-coder:7b..."
    SMOKE=$(ollama run qwen2.5-coder:7b "respond with only the word: ok" --nowordwrap 2>/dev/null | tr -d '[:space:]')
    if echo "$SMOKE" | grep -qi "ok"; then
      ok "Smoke test passed"
    else
      warn "Smoke test response: '$SMOKE' — may still be fine, continuing..."
    fi
  fi

  case "$NODE_TYPE" in
    nvidia)
      log "Pulling mid-tier models (NVIDIA ~40GB VRAM)..."
      for m in "${MID_MODELS[@]}"; do pull_if_missing "$m"; done
      ;;
    strix|gb10|unified)
      log "Pulling mid-tier models..."
      for m in "${MID_MODELS[@]}"; do pull_if_missing "$m"; done
      log "Pulling large models (128GB unified memory)..."
      log "Note: ~250GB total — this will take a while on first run"
      for m in "${LARGE_MODELS[@]}"; do pull_if_missing "$m"; done
      ;;
    *)
      warn "Unknown node type — only common models pulled"
      ;;
  esac
fi

# ── Open WebUI (optional) ─────────────────────────────────────────────────────
section "Open WebUI"

if $WITH_WEBUI; then
  if ! command -v docker &>/dev/null; then
    warn "Docker not installed — skipping Open WebUI"
    warn "Install: https://docs.docker.com/engine/install/ubuntu/"
  else
    if sudo docker ps --format "{{.Names}}" 2>/dev/null | grep -q "^open-webui$"; then
      skip "Open WebUI already running"
    else
      log "Deploying Open WebUI (standalone UI, native Ollama backend)..."
      run sudo docker run -d \
        --name open-webui \
        --restart always \
        -p 8080:8080 \
        -v open-webui:/app/backend/data \
        -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
        --add-host=host.docker.internal:host-gateway \
        ghcr.io/open-webui/open-webui:main
      ok "Open WebUI deployed → http://$(hostname -I | awk '{print $1}'):8080"
    fi
  fi
else
  skip "Open WebUI not requested (pass --with-webui to deploy)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
section "Done"

echo ""
echo "  Node:      $NODE_NAME ($NODE_TYPE)"
echo "  Arch:      $ARCH"
echo "  Memory:    ${TOTAL_MEM_GB}GB"
if ! $DRY_RUN; then
  MODEL_COUNT=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
  echo "  Models:    $MODEL_COUNT pulled"
fi
echo ""

if [ -f config/nodes.yaml ]; then
  NEEDS_IP=$(grep -c "192.168.1.XX" config/nodes.yaml 2>/dev/null || echo 0)
  if [ "$NEEDS_IP" -gt 0 ]; then
    warn "config/nodes.yaml still has placeholder IPs — update before benchmarking"
  fi
fi

echo "  Next steps:"
echo "    Edit config/nodes.yaml with node IPs"
echo "    python scripts/benchmark.py --node $NODE_NAME --suite standard"
echo "    python scripts/report_html.py --results results/"
echo ""
echo "  Re-run options:"
echo "    ./bootstrap.sh --skip-models           # config/Ollama update only"
echo "    ./bootstrap.sh --with-webui            # add Open WebUI"
echo "    ./bootstrap.sh --teardown-docker       # remove Docker Ollama first"
echo ""

if $DRY_RUN; then
  warn "DRY RUN — no changes were made"
fi
