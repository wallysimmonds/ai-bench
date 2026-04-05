#!/bin/bash
# setup-nvidia.sh — Full automated setup for nvidia-ai (RTX 5070 Ti + RTX PRO 4000 Blackwell)
# Usage: ./scripts/setup/setup-nvidia.sh [host] [user]
# Prerequisites: SSH key must be in authorized_keys on the target node.
#   To add it: ssh-copy-id -i ~/.ssh/id_ed25519 johanus@192.168.1.211
set -euo pipefail

HOST=${1:-"192.168.1.211"}
REMOTE_USER=${2:-"johanus"}
SSH="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ${REMOTE_USER}@${HOST}"
OLLAMA_MODELS_DIR="/home/${REMOTE_USER}/ollama-models"

echo "=== Setting up nvidia-ai: ${REMOTE_USER}@${HOST} ==="

# ── 1. System info ────────────────────────────────────────────────────────────
echo "--- System info ---"
$SSH "uname -m && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable'"

# ── 2. Install Ollama ─────────────────────────────────────────────────────────
echo "--- Installing / verifying Ollama ---"
$SSH "
  if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable ollama
  fi
  echo \"Ollama: \$(ollama --version)\"
"

# ── 3. Configure Ollama service ───────────────────────────────────────────────
echo "--- Configuring Ollama service ---"
$SSH "
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null << 'EOF'
[Service]
Environment=\"OLLAMA_MODELS=${OLLAMA_MODELS_DIR}\"
Environment=\"OLLAMA_HOST=0.0.0.0:11434\"
EOF
  mkdir -p ${OLLAMA_MODELS_DIR}
  sudo systemctl daemon-reload
  sudo systemctl restart ollama
  sleep 3
  systemctl is-active ollama
"

# ── 4. Pull Ollama-native models ───────────────────────────────────────────────
echo "--- Pulling models (40GB VRAM) ---"

ollama_pull() {
  local tag=$1
  echo "  Pulling ${tag}..."
  $SSH "
    if ollama list | grep -q '^${tag} '; then
      echo '  ${tag}: already present, skipping'
    else
      ollama pull ${tag}
    fi
  "
}

# Common
ollama_pull qwen2.5-coder:7b
ollama_pull qwen2.5-coder:14b
ollama_pull qwen3.5:9b

# Mid-tier — fits in 40GB dual GPU
ollama_pull qwen2.5-coder:32b
ollama_pull qwen3.5:35b-a3b

# ── 5. Smoke test ─────────────────────────────────────────────────────────────
echo "--- Smoke test ---"
$SSH "ollama run qwen2.5-coder:7b 'respond with only: ok' --nowordwrap 2>/dev/null"

# ── 6. Final model list ───────────────────────────────────────────────────────
echo "--- Final model list ---"
$SSH "ollama list"

echo ""
echo "=== nvidia-ai setup complete ==="
