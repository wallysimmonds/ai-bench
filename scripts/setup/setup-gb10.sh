#!/bin/bash
# setup-gb10.sh — Full automated setup for lenovo-gb10 (Grace Blackwell GB10)
# Usage: ./scripts/setup/setup-gb10.sh [host] [user]
# Prerequisites: SSH key must be in authorized_keys on the target node.
#   To add it: ssh-copy-id -i ~/.ssh/id_ed25519 wally@192.168.1.52
set -euo pipefail

HOST=${1:-"192.168.1.52"}
REMOTE_USER=${2:-"wally"}
SSH="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ${REMOTE_USER}@${HOST}"
HF_MODELS_DIR="/home/${REMOTE_USER}/models"
OLLAMA_MODELS_DIR="/home/${REMOTE_USER}/ollama-models"

echo "=== Setting up lenovo-gb10: ${REMOTE_USER}@${HOST} ==="

# ── 1. System info ────────────────────────────────────────────────────────────
echo "--- System info ---"
$SSH "uname -m && uname -r && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable'"

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

# ── 4. Install huggingface-cli (needed for models not on Ollama registry) ─────
echo "--- Installing huggingface-cli ---"
$SSH "
  if ! python3 -c 'import huggingface_hub' &>/dev/null; then
    pip3 install huggingface_hub --break-system-packages -q
  fi
  python3 -c 'import huggingface_hub; print(\"huggingface_hub:\", huggingface_hub.__version__)'
"

# ── 5. Pull Ollama-native models ───────────────────────────────────────────────
echo "--- Pulling Ollama-native models ---"

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

ollama_pull qwen2.5-coder:7b
ollama_pull qwen3.5:9b
ollama_pull qwen3.5:35b-a3b
ollama_pull deepseek-r1:70b
ollama_pull llama3.3:70b
ollama_pull qwen3-coder-next:latest
ollama_pull gpt-oss:120b
ollama_pull qwen3.5:122b

# ── 6. HuggingFace models (not available on Ollama registry) ──────────────────
echo "--- Pulling HuggingFace models ---"

hf_pull_and_import() {
  local ollama_tag=$1
  local hf_repo=$2
  local hf_file=$3
  local local_dir="${HF_MODELS_DIR}/$(echo ${ollama_tag} | tr ':' '-')"

  echo "  ${ollama_tag} (from ${hf_repo})..."
  $SSH "
    # Skip if already in Ollama
    if ollama list | grep -q '^${ollama_tag} '; then
      echo '  ${ollama_tag}: already present, skipping'
      exit 0
    fi

    mkdir -p ${local_dir}

    # Download GGUF if not already on disk
    if [ ! -f '${local_dir}/${hf_file}' ]; then
      echo '  Downloading ${hf_file}...'
      ~/.local/bin/hf download ${hf_repo} --include '${hf_file}' --local-dir ${local_dir}
    else
      echo '  ${hf_file}: already on disk'
    fi

    # Import into Ollama
    echo 'FROM ${local_dir}/${hf_file}' > /tmp/Modelfile.tmp
    ollama create ${ollama_tag} -f /tmp/Modelfile.tmp
    rm /tmp/Modelfile.tmp
    echo '  ${ollama_tag}: imported successfully'
  "
}

hf_pull_and_import "qwen3.5:27b" \
  "bartowski/Qwen_Qwen3.5-27B-GGUF" \
  "Qwen_Qwen3.5-27B-Q4_K_M.gguf"

# ── 7. Smoke test ─────────────────────────────────────────────────────────────
echo "--- Smoke test ---"
$SSH "ollama run qwen2.5-coder:7b 'respond with only: ok' --nowordwrap 2>/dev/null"

# ── 8. Final model list ───────────────────────────────────────────────────────
echo "--- Final model list ---"
$SSH "ollama list"

echo ""
echo "=== lenovo-gb10 setup complete ==="
