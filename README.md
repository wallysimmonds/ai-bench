# ai-bench

Local AI inference benchmarking suite — measures tokens/sec, TTFT, and prompt throughput across a heterogeneous fleet using two complementary test methods.

## Fleet

| Node | Hardware | Memory | Architecture |
|------|----------|--------|--------------|
| `lenovo-gb10` | Grace Blackwell GB10 | 128GB unified | ARM, Ubuntu + Nvidia kernel |
| `bosgame-m5` | AMD AI Max 395 (Strix Halo) | 128GB DDR5 unified | x86, ROCm gfx1151 |
| `nvidia-ai` | RTX 5070 Ti 16GB + RTX PRO 4000 Blackwell 24GB | 40GB VRAM | x86, dual GPU |

See `config/nodes.yaml` for SSH config and `config/models.yaml` for per-node model assignments.

## Benchmark Types

### Ollama benchmark (`scripts/benchmark.py`)
Tests the full inference stack via the Ollama HTTP API. Measures:
- **TG tok/s** — text generation (decode) throughput
- **PP tok/s** — prompt processing (prefill) throughput
- **TTFT** — time to first token

### llama-bench (`scripts/llama_bench.sh`)
Runs llama.cpp's `llama-bench` directly against GGUF files, bypassing Ollama entirely. Uses standardised pp512/tg128 batch sizes for community-comparable numbers. Typically shows ~50% higher PP throughput than Ollama due to larger batch sizes and no HTTP overhead.

## Quick Start

```bash
# 1. Clone
git clone <repo> ai-bench && cd ai-bench

# 2. Set up a node (first-time only — see prerequisites below)
./scripts/setup/setup-gb10.sh [host] [user]
./scripts/setup/setup-strix.sh [host] [user]

# 3. Run full benchmark suite against a node
./scripts/run_all.sh lenovo-gb10
./scripts/run_all.sh bosgame-m5

# 4. Generate HTML report
python3 scripts/report_html.py
# → reports/report.html
```

`run_all.sh` runs both benchmark types sequentially for each model (memory constraint — only one model loaded at a time), unloading Ollama between runs.

## Prerequisites

### lenovo-gb10 (Grace Blackwell GB10)

Manual steps before running `setup-gb10.sh`:
1. Ubuntu + Nvidia GB10 kernel installed
2. SSH key authorised: `ssh-copy-id -i ~/.ssh/id_ed25519.pub wally@<host>`
3. Passwordless sudo: `echo "wally ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/wally-nopasswd`

The setup script handles: Ollama install, service config, huggingface-cli, model pulls, llama.cpp CUDA build, GGUF symlinks.

### bosgame-m5 (AMD AI Max 395 / Strix Halo)

Manual steps before running `setup-strix.sh`:
1. Ubuntu 25.10 server install (NOT desktop, NOT encrypted)
2. BIOS: Secure Boot disabled, GART/UMA Frame Buffer Size = 512MB
3. Kernel 6.18.9+ via mainline:
   ```bash
   sudo add-apt-repository ppa:cappelikan/ppa -y
   sudo apt update && sudo apt install -y mainline pkexec
   sudo mainline install 6.18.9 && sudo reboot
   ```
4. TTM config for 124GB GPU memory:
   ```bash
   sudo tee /etc/modprobe.d/amdgpu-llm.conf << 'EOF'
   options ttm pages_limit=32505856
   options ttm page_pool_size=32505856
   EOF
   sudo update-initramfs -u && sudo reboot
   ```
5. ROCm via amdgpu-install:
   ```bash
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb
   sudo apt install -y ./amdgpu-install_7.2.1.70201-1_all.deb
   sudo amdgpu-install -y --usecase=rocm --no-dkms
   sudo usermod -aG render,video $USER && sudo reboot
   ```
6. SSH key authorised, passwordless sudo (see setup-strix.sh header)

The setup script handles: pre-flight checks (kernel, GTT, ROCm), Ollama install + ROCm config, model pulls, llama.cpp ROCm/HIP build, GGUF symlinks.

Verify prerequisites:
```bash
uname -r                     # should be 6.18.x
rocminfo | grep gfx          # should show gfx1151
echo "scale=1; $(cat /sys/class/drm/card1/device/mem_info_gtt_total) / 1024^3" | bc  # ~124
```

## Repo Structure

```
ai-bench/
├── config/
│   ├── nodes.yaml          # Node definitions (host, user, SSH key, GPU specs)
│   └── models.yaml         # Model list with arch, size, source, per-node assignments
├── scripts/
│   ├── setup/
│   │   ├── setup-gb10.sh   # Full setup for Grace Blackwell GB10 (CUDA)
│   │   └── setup-strix.sh  # Full setup for AMD Strix Halo (ROCm)
│   ├── benchmark.py        # Ollama HTTP benchmark runner
│   ├── llama_bench.sh      # llama-bench wrapper (direct GGUF)
│   ├── run_all.sh          # Orchestrator: runs both benchmarks per model
│   └── report_html.py      # HTML report generator
├── results/                # Raw JSON results per run (gitignored)
└── reports/                # Generated HTML reports (gitignored)
```

## Known Limitations

- **Ollama qwen3.5 GGUFs skip llama-bench**: All Ollama-packaged qwen3.5 GGUFs (9b, 35b-a3b, 122b) fail with `rope.dimension_sections has wrong array length; expected 4, got 3`. HuggingFace-sourced GGUFs (e.g. bartowski qwen3.5:27b) are compatible. Ollama benchmark numbers are unaffected.
- **gpt-oss:120b skips llama-bench**: Ollama GGUF uses different tensor names than llama.cpp's OPENAI_MOE implementation. HF-sourced GGUFs would work.
- **ROCm shader cache warmup**: First llama-bench run on Strix Halo compiles shaders — subsequent runs are faster. Results from first run may be unrepresentative.
- **Sequential model loading**: Nodes with 128GB unified memory can only run one model at a time. `run_all.sh` enforces this by unloading Ollama between models.
- **MoE TG vs PP**: MoE models (e.g. gpt-oss:120b, qwen3.5:122b) show higher TG tok/s than equivalent-size dense models because only a fraction of parameters activate per token. The report annotates all models as dense or MoE.
