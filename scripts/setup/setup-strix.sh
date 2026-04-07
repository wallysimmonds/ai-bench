#!/bin/bash
# setup-strix.sh — Full automated setup for bosgame-m5 (AMD AI Max+ 395 Strix Halo)
#
# Prerequisites (manual, one-time before running this script):
#   1. Ubuntu 25.10 server install (NOT desktop, NOT encrypted)
#   2. BIOS settings:
#        - Secure Boot: Disabled
#        - GART / UMA Frame Buffer Size: 512MB
#   3. Kernel 6.18.9+ via mainline:
#        sudo add-apt-repository ppa:cappelikan/ppa -y
#        sudo apt update && sudo apt install -y mainline pkexec
#        sudo mainline install 6.18.9
#        sudo reboot
#   4. TTM config for 124GB GPU memory:
#        sudo tee /etc/modprobe.d/amdgpu-llm.conf << 'EOF'
#        options ttm pages_limit=32505856
#        options ttm page_pool_size=32505856
#        EOF
#        sudo update-initramfs -u && sudo reboot
#   5. ROCm via amdgpu-install:
#        wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
#        sudo apt install -y ./amdgpu-install_7.2.1.70201-1_all.deb
#        sudo amdgpu-install -y --usecase=rocm --no-dkms
#        sudo usermod -aG render,video $USER
#        sudo reboot
#   6. SSH key authorised:
#        ssh-copy-id -i ~/.ssh/id_ed25519.pub johanus@<host>
#   7. Passwordless sudo:
#        echo "johanus ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/johanus-nopasswd
#
# Verify prerequisites with:
#   uname -r                                         # should be 6.18.x
#   rocminfo | grep gfx                              # should show gfx1151
#   echo "scale=1; $(cat /sys/class/drm/card1/device/mem_info_gtt_total) / 1024^3" | bc  # ~124
#
# Usage:
#   ./scripts/setup/setup-strix.sh                       # uses defaults
#   ./scripts/setup/setup-strix.sh 192.168.1.54 johanus  # explicit host/user
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HOST=${1:-"192.168.1.54"}
REMOTE_USER=${2:-"johanus"}
SSH="ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no ${REMOTE_USER}@${HOST}"
HF_MODELS_DIR="/home/${REMOTE_USER}/models"
LLAMA_DIR="/home/${REMOTE_USER}/llama.cpp"

echo "=== Setting up bosgame-m5 (Strix Halo): ${REMOTE_USER}@${HOST} ==="

# ── 0. Pre-flight check ───────────────────────────────────────────────────────
echo "--- Pre-flight check ---"
if ! $SSH "echo 'SSH OK' && sudo true && echo 'Sudo OK'"; then
  echo "ERROR: SSH or sudo failed. Check prerequisites above."
  exit 1
fi

KERNEL=$($SSH "uname -r")
echo "  Kernel: ${KERNEL}"
if [[ "${KERNEL}" < "6.18" ]]; then
  echo "ERROR: Kernel ${KERNEL} is below 6.18 — see prerequisites for mainline install."
  exit 1
fi

GTT=$($SSH "cat /sys/class/drm/card1/device/mem_info_gtt_total 2>/dev/null || echo 0")
GTT_GB=$(echo "scale=0; ${GTT} / 1024 / 1024 / 1024" | bc)
echo "  GTT: ${GTT_GB}GB"
if [ "${GTT_GB}" -lt 100 ]; then
  echo "ERROR: GTT is only ${GTT_GB}GB — TTM config may not have applied. See prerequisites."
  exit 1
fi

DISK_FREE=$($SSH "df -BG / | tail -1 | awk '{print \$4}' | tr -d G")
echo "  Disk free: ${DISK_FREE}GB"
if [ "${DISK_FREE}" -lt 400 ]; then
  echo "WARNING: Only ${DISK_FREE}GB free. Large model set requires ~400GB."
fi

$SSH "rocminfo 2>/dev/null | grep -q gfx1151 && echo '  ROCm: gfx1151 OK' || echo 'WARNING: gfx1151 not detected by ROCm'"

# ── 1. System info ────────────────────────────────────────────────────────────
echo "--- System info ---"
$SSH "uname -m && uname -r && rocminfo 2>/dev/null | grep 'Marketing Name' | head -2"

# ── 2. Install Ollama ─────────────────────────────────────────────────────────
echo "--- Installing / verifying Ollama ---"
$SSH "
  if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable ollama
  fi
  echo \"Ollama: \$(ollama --version)\"
"

# ── 3. Configure Ollama service (AMD ROCm) ────────────────────────────────────
echo "--- Configuring Ollama service ---"
$SSH "
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null << 'EOF'
[Service]
Environment=\"OLLAMA_HOST=0.0.0.0:11434\"
Environment=\"HSA_OVERRIDE_GFX_VERSION=11.0.0\"
Environment=\"ROCR_VISIBLE_DEVICES=0\"
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
ollama_pull qwen3.5:9b
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
      ~/.local/bin/huggingface-cli download ${hf_repo} --include '${hf_file}' --local-dir ${local_dir}
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

# ── 7. Build llama.cpp (ROCm/HIP) ─────────────────────────────────────────────
echo "--- Building llama.cpp with ROCm ---"
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
    cmake -B build \
      -DGGML_HIPBLAS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DAMDGPU_TARGETS=gfx1151 \
      -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build build --config Release -j\$(nproc) --target llama-bench llama-cli
    echo \"llama-bench built OK\"
  fi
"

# ── 8. Create GGUF symlinks for llama-bench ───────────────────────────────────
echo "--- Setting up llama-bench GGUF symlinks ---"
$SSH "
  for tag in 'qwen2.5-coder:7b' 'qwen2.5-coder:14b' 'qwen3.5:9b' 'qwen3.5:35b-a3b' 'gemma4:26b' 'deepseek-r1:70b' 'llama3.3:70b' 'qwen3-coder-next:latest' 'gpt-oss:120b' 'qwen3.5:122b'; do
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
$SSH "ollama run qwen3.5:9b 'respond with only: ok' --nowordwrap 2>/dev/null"

# ── 10. Final model list ──────────────────────────────────────────────────────
echo "--- Final model list ---"
$SSH "ollama list"

echo ""
echo "=== bosgame-m5 setup complete ==="
echo "Next: ./scripts/run_all.sh bosgame-m5"
