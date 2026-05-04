# AI Fleet Benchmark Agent Guide

You are operating on the **bosgame-m5** node (AMD Strix Halo, 192.168.1.54). The project is at `~/ai-bench/`.

---

## STOP CONDITIONS — Do not attempt workarounds, ask the user instead

- **SSH auth failure / Permission denied** — All nodes require key-based auth. `~/.ssh/id_ed25519` must be in `authorized_keys` on the target node. Never attempt sshpass, expect, or password workarounds. Ask the user to run `ssh-copy-id` from their local machine.
- **Node unreachable** — Check cables/power, ask the user. Do not loop on retries.
- **Model not loaded in Ollama** — Run `ollama pull <tag>` and wait. If it fails, ask the user.

## Node Prerequisites

Before benchmarking a node for the first time, verify these are in place. If not, fix them — you have the access to do so:

**SSH key auth** — from this machine (bosgame-m5):
```bash
# Generate key if needed
test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
# Install key on target node (requires password once)
ssh-copy-id -i ~/.ssh/id_ed25519 <user>@<host>
```

**Passwordless sudo** — required on all nodes for package installation and setup:
```bash
ssh <user>@<host> "sudo grep -q 'NOPASSWD' /etc/sudoers || echo '<user> ALL=(ALL) NOPASSWD: ALL' | sudo tee -a /etc/sudoers"
```
If that fails due to missing sudo, ask the user to run `sudo visudo` and add `<user> ALL=(ALL) NOPASSWD: ALL`.

**Ollama installed and running:**
```bash
ssh <user>@<host> "curl -s http://localhost:11434/api/tags > /dev/null && echo 'ollama ok' || echo 'ollama not running'"
```
If not running: `ssh <user>@<host> "curl -fsSL https://ollama.com/install.sh | sh"`

**Run preflight check before any benchmark:**
```bash
ssh -i ~/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=5 <user>@<host> "echo ok" || { echo "SSH failed — set up key auth first"; exit 1; }
curl -s --max-time 5 http://<host>:11434/api/tags > /dev/null || { echo "Ollama not reachable"; exit 1; }
```

---

## Fleet Nodes

| Name | IP | User | Hardware |
|------|----|------|----------|
| `bosgame-m5` | 192.168.1.54 | johanus | AMD AI Max 395 (Strix Halo), 128GB, ROCm |
| `lenovo-gb10` | 192.168.1.52 | wally | Grace Blackwell GB10, 128GB, CUDA |
| `nvidia-ai` | 192.168.1.211 | johanus | RTX 5070 Ti + RTX PRO 4K, 40GB VRAM |
| `amd-ai` | 192.168.1.80 | johanus | AMD Ryzen 7 8700G + RX 9070 (gfx1200) 16GB, 48GB RAM, ROCm/RDNA4 |

SSH key for all nodes: `~/.ssh/id_ed25519`

**Test SSH before doing anything else:**
```bash
ssh -i ~/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=5 <user>@<host> "echo ok"
```
If this fails, stop and tell the user — do not attempt alternatives.

---

## Project Structure

```
~/ai-bench/
├── config/
│   ├── nodes.yaml       # Node definitions (IPs, users, SSH keys)
│   └── models.yaml      # Model assignments per node
├── scripts/
│   ├── run_all.sh       # Main orchestrator — Ollama + llama-bench
│   ├── benchmark.py     # Ollama HTTP benchmark runner
│   ├── llama_bench.sh   # llama-bench wrapper
│   ├── report_html.py   # HTML report generator
│   └── report_excel.py  # Excel report generator
├── results/             # JSON benchmark results
├── reports/             # Generated HTML/Excel reports
└── docs/runbook.md      # Full operations reference
```

---

## Common Tasks

### Run benchmarks on a node
```bash
cd ~/ai-bench
./scripts/run_all.sh bosgame-m5           # full run (Ollama + llama-bench)
./scripts/run_all.sh lenovo-gb10 --skip-llama-bench  # Ollama only
./scripts/run_all.sh bosgame-m5 --models "qwen3.5:35b-a3b"  # single model
```

### ALWAYS warm up the Strix first after idle (ROCm shader compilation)
```bash
./scripts/run_all.sh bosgame-m5 --models "qwen2.5-coder:7b" --skip-llama-bench
```

### Context scaling benchmarks
```bash
bash scripts/llama_bench.sh bosgame-m5 --contexts "500,2048,8192,16384,28672"
```

### Generate HTML report
```bash
python3 scripts/report_html.py --results results/
# Opens: reports/benchmark_<timestamp>.html
```

### Check Ollama health on a node
```bash
curl -s http://192.168.1.54:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']), 'models loaded')"
```

### Check thermals
```bash
ssh johanus@192.168.1.54 "sensors"   # Strix
ssh wally@192.168.1.52 "sensors"     # GB10
```

---

## Downloading GGUFs for llama-bench

Ollama's internal model blobs are not usable by llama-bench. Use `hf` CLI:
```bash
mkdir -p ~/models/<slug>
~/.local/bin/hf download <owner>/<repo> --include '<file>.gguf' --local-dir ~/models/<slug>
```

After downloading, update `config/models.yaml` to set `source: hf` with `hf_repo` and `hf_file`.

---

## Key Behaviours

- **Run long commands in the background** — benchmarks take 30-90 mins. Use background execution and check progress periodically rather than blocking.
- **One model at a time on unified memory nodes** — bosgame-m5 and lenovo-gb10 unload each model between tests automatically via run_all.sh.
- **Results accumulate** — all JSON files in `results/` are loaded by the report generator. You don't need to clear old results.
- **ROCm on Strix** — llama-bench uses HIP/ROCm. The build at `~/llama.cpp/build/bin/llama-bench` was compiled with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151`. If it falls back to CPU (check `backends` field in results JSON), rebuild with `bash scripts/llama_bench.sh bosgame-m5 --install`.
- **Git** — this project is at `https://github.com/wallysimmonds/ai-bench`. Pull updates with `git pull`. `config/nodes.yaml` is gitignored — do not expect it to be in the repo.
