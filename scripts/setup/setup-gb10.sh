#!/bin/bash
# setup-gb10.sh — Full automated setup for lenovo-gb10 (Grace Blackwell GB10)
#
# Prerequisites (manual, one-time before running this script):
#   1. Ubuntu + Nvidia GB10 kernel installed on the node
#   2. SSH key authorised: ssh-copy-id -i ~/.ssh/id_ed25519.pub wally@<host>
#   3. Passwordless sudo:
#        echo "wally ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/wally-nopasswd
#
# Usage:
#   ./scripts/setup/setup-gb10.sh                    # uses defaults from config/nodes.yaml
#   ./scripts/setup/setup-gb10.sh 192.168.1.52 wally # explicit host/user
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HOST=${1:-"192.168.1.52"}
REMOTE_USER=${2:-"wally"}
SSH="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ${REMOTE_USER}@${HOST}"
HF_MODELS_DIR="/home/${REMOTE_USER}/models"
LLAMA_DIR="/home/${REMOTE_USER}/llama.cpp"

echo "=== Setting up lenovo-gb10: ${REMOTE_USER}@${HOST} ==="

# ── 0. Pre-flight check ───────────────────────────────────────────────────────
echo "--- Pre-flight check ---"
if ! $SSH "echo 'SSH OK' && sudo true && echo 'Sudo OK'"; then
  echo "ERROR: SSH or sudo failed. Check prerequisites above."
  exit 1
fi

DISK_FREE=$($SSH "df -BG / | tail -1 | awk '{print \$4}' | tr -d G")
if [ "${DISK_FREE}" -lt 200 ]; then
  echo "WARNING: Only ${DISK_FREE}GB free on /. Models may not all fit."
fi

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
Environment=\"OLLAMA_HOST=0.0.0.0:11434\"
EOF
  sudo systemctl daemon-reload
  sudo systemctl restart ollama
  sleep 3
  systemctl is-active ollama
"

# ── 4. Install huggingface-cli ────────────────────────────────────────────────
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
ollama_pull qwen2.5-coder:14b
ollama_pull qwen2.5-coder:32b
ollama_pull qwen3.5:9b
ollama_pull qwen3-coder:30b
ollama_pull qwen3.5:35b-a3b
ollama_pull gemma4:26b
ollama_pull deepseek-r1:70b
ollama_pull llama3.3:70b
ollama_pull qwen3-coder-next:latest
ollama_pull gpt-oss:120b
ollama_pull qwen3.5:122b

# ── 6. HuggingFace models ─────────────────────────────────────────────────────
echo "--- Pulling HuggingFace models ---"

hf_pull_and_import() {
  local ollama_tag=$1
  local hf_repo=$2
  local hf_file=$3
  local renderer=${4:-""}
  local local_dir="${HF_MODELS_DIR}/$(echo ${ollama_tag} | tr ':' '-')"

  echo "  ${ollama_tag} (from ${hf_repo})..."
  $SSH "
    if ollama list | grep -q '^${ollama_tag} '; then
      echo '  ${ollama_tag}: already present, skipping'
      exit 0
    fi

    mkdir -p ${local_dir}

    if [ ! -f '${local_dir}/${hf_file}' ]; then
      echo '  Downloading ${hf_file}...'
      ~/.local/bin/hf download ${hf_repo} --include '${hf_file}' --local-dir ${local_dir}
    else
      echo '  ${hf_file}: already on disk'
    fi

    {
      echo 'FROM ${local_dir}/${hf_file}'
      if [ -n '${renderer}' ]; then
        echo 'RENDERER ${renderer}'
        echo 'PARSER ${renderer}'
        echo 'PARAMETER presence_penalty 1.5'
        echo 'PARAMETER temperature 1'
        echo 'PARAMETER top_k 20'
        echo 'PARAMETER top_p 0.95'
      fi
    } > /tmp/Modelfile.tmp
    ollama create ${ollama_tag} -f /tmp/Modelfile.tmp
    rm /tmp/Modelfile.tmp
    echo '  ${ollama_tag}: imported successfully'
  "
}

hf_pull_and_import "qwen3.5:27b" \
  "bartowski/Qwen_Qwen3.5-27B-GGUF" \
  "Qwen_Qwen3.5-27B-Q4_K_M.gguf"

# ── 7. Build llama.cpp (CUDA) ─────────────────────────────────────────────────
echo "--- Building llama.cpp with CUDA ---"
$SSH "
  if [ -f ${LLAMA_DIR}/build/bin/llama-bench ]; then
    echo 'llama-bench already built, skipping'
  else
    sudo apt-get install -y cmake build-essential git libcurl4-openssl-dev 2>/dev/null
    if [ ! -d ${LLAMA_DIR} ]; then
      git clone --depth 1 https://github.com/ggerganov/llama.cpp ${LLAMA_DIR}
    else
      cd ${LLAMA_DIR} && git pull
    fi
    cd ${LLAMA_DIR}

    # Find nvcc — not always on PATH on GB10
    NVCC=\$(find /usr/local/cuda* -name nvcc 2>/dev/null | head -1)
    if [ -z \"\${NVCC}\" ]; then
      echo 'WARNING: nvcc not found, building CPU-only'
      cmake -B build -DCMAKE_BUILD_TYPE=Release
    else
      echo \"Using nvcc: \${NVCC}\"
      cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=\${NVCC}
    fi

    cmake --build build --config Release -j\$(nproc) --target llama-bench llama-cli
    echo \"llama-bench built OK\"
  fi
"

# ── 8. Create GGUF symlinks for llama-bench ───────────────────────────────────
echo "--- Setting up llama-bench GGUF symlinks ---"
$SSH "
  BLOB_DIR=\$(ollama show qwen2.5-coder:7b --modelfile 2>/dev/null | grep '^FROM' | awk '{print \$2}' | xargs dirname 2>/dev/null || echo '')
  if [ -z \"\${BLOB_DIR}\" ]; then
    echo 'Could not determine Ollama blob dir — skipping symlinks'
    exit 0
  fi
  echo \"Ollama blob dir: \${BLOB_DIR}\"

  # Create symlinks for all Ollama-sourced models
  for tag in 'qwen2.5-coder:7b' 'qwen2.5-coder:14b' 'qwen2.5-coder:32b' 'qwen3.5:9b' 'qwen3-coder:30b' 'qwen3.5:35b-a3b' 'gemma4:26b' 'deepseek-r1:70b' 'llama3.3:70b' 'qwen3-coder-next:latest' 'gpt-oss:120b' 'qwen3.5:122b'; do
    blob=\$(ollama show \"\${tag}\" --modelfile 2>/dev/null | grep '^FROM' | awk '{print \$2}')
    slug=\$(echo \"\${tag}\" | tr ':' '-')
    if [ -n \"\${blob}\" ] && [ -f \"\${blob}\" ]; then
      mkdir -p ${HF_MODELS_DIR}/\${slug}
      ln -sf \"\${blob}\" ${HF_MODELS_DIR}/\${slug}/\${slug}.gguf
      echo \"  OK \${slug}\"
    else
      echo \"  SKIP \${tag} (not pulled yet)\"
    fi
  done
"

# ── 9. Smoke test ─────────────────────────────────────────────────────────────
echo "--- Smoke test ---"
$SSH "ollama run qwen2.5-coder:7b 'respond with only: ok' --nowordwrap 2>/dev/null"

# ── 10. Final model list ──────────────────────────────────────────────────────
echo "--- Final model list ---"
$SSH "ollama list"

echo ""
echo "=== lenovo-gb10 setup complete ==="
echo "Next: ./scripts/run_all.sh lenovo-gb10"
