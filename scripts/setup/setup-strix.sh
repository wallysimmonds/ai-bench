#!/bin/bash
# setup-strix.sh — Ollama install + model pull for bosgame-m5 (Strix Halo)
# Usage: ./scripts/setup/setup-strix.sh [host] [user]

HOST=${1:-"192.168.1.XX"}
USER=${2:-"brendan"}

echo "=== Setting up bosgame-m5 (Strix Halo): $USER@$HOST ==="

ssh $USER@$HOST << 'REMOTE'
set -e

echo "--- Installing Ollama ---"
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    sudo systemctl enable --now ollama
    sleep 3
fi

echo "Ollama version: $(ollama --version)"

# Strix Halo — AMD unified memory, set ROCm hints
sudo mkdir -p /etc/systemd/system/ollama.service.d
cat << 'EOF' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_MODELS=/home/brendan/ollama-models"
Environment="OLLAMA_HOST=0.0.0.0:11434"
# AMD GPU — ROCm
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 3

mkdir -p ~/ollama-models

echo "--- Pulling models for bosgame-m5 (128GB unified) ---"

# Common
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull qwen3.5:9b

# Mid-tier
ollama pull qwen2.5-coder:32b
ollama pull qwen3.5:27b
ollama pull qwen3.5:35b-a3b

# Large — this is the Strix's advantage
ollama pull qwen3.5:72b
ollama pull qwen3-coder-next:80b-a3b-q4_K_M   # Key agentic benchmark
# ollama pull qwen3.5:122b-a10b-q4_K_M        # Uncomment if you have headroom

echo "--- Model list ---"
ollama list

echo "--- Quick smoke test ---"
ollama run qwen2.5-coder:7b "respond with only: ok" --nowordwrap

echo "=== bosgame-m5 setup complete ==="
REMOTE
