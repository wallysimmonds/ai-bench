# AI Fleet Benchmark Runbook

Common commands for running, managing, and extending benchmarks across the fleet.

---

## Fleet Nodes

| Name | IP | User | Hardware | SSH Key |
|------|----|------|----------|---------|
| `bosgame-m5` | 192.168.1.54 | johanus | AMD AI Max 395 (Strix Halo), 128GB unified, ROCm | ~/.ssh/id_ed25519 |
| `lenovo-gb10` | 192.168.1.52 | wally | Grace Blackwell GB10, 128GB unified, CUDA | ~/.ssh/id_ed25519 |
| `nvidia-ai` | 192.168.1.211 | johanus | RTX 5070 Ti + RTX PRO 4K, 40GB VRAM, CUDA | ~/.ssh/id_ed25519 |

---

## Running Benchmarks

### Full run — Ollama + llama-bench (recommended)
```bash
./scripts/run_all.sh bosgame-m5
./scripts/run_all.sh lenovo-gb10
```

### Strix warmup — always do this first after idle (ROCm shader compilation)
```bash
./scripts/run_all.sh bosgame-m5 --models "qwen2.5-coder:7b" --skip-llama-bench
./scripts/run_all.sh bosgame-m5
```

### Both nodes in parallel
```bash
./scripts/run_all.sh bosgame-m5 &
./scripts/run_all.sh lenovo-gb10 &
wait
```

### Ollama benchmarks only (skip llama-bench)
```bash
./scripts/run_all.sh bosgame-m5 --skip-llama-bench
```

### Specific models only
```bash
./scripts/run_all.sh bosgame-m5 --models "qwen3.5:35b-a3b,qwen3.5:122b"
```

### llama-bench only (no Ollama)
```bash
bash scripts/llama_bench.sh bosgame-m5
bash scripts/llama_bench.sh lenovo-gb10
```

### llama-bench with context scaling
```bash
# Community standard (pp512)
bash scripts/llama_bench.sh bosgame-m5

# Context scaling — measures PP throughput at multiple context sizes
bash scripts/llama_bench.sh bosgame-m5 --contexts "500,2048,8192,16384,28672"
bash scripts/llama_bench.sh lenovo-gb10 --contexts "500,2048,8192,16384,28672"
```

### Specific models with context scaling
```bash
bash scripts/llama_bench.sh bosgame-m5 --models "qwen3.5:35b-a3b" --contexts "500,2048,8192,16384,28672"
```

---

## Generating Reports

### HTML report (from all results in results/)
```bash
python3 scripts/report_html.py --results results/
# Output: reports/benchmark_<timestamp>.html
```

### Excel report
```bash
python3 scripts/report_excel.py --results results/
# Output: reports/benchmark_<timestamp>.xlsx
```

---

## llama-bench Setup

### First-time install / rebuild on a node
```bash
bash scripts/llama_bench.sh bosgame-m5 --install
bash scripts/llama_bench.sh lenovo-gb10 --install
```

The script auto-detects the backend:
- CUDA if `nvcc` or `/usr/local/cuda` found
- ROCm/HIP if `hipcc` or `/opt/rocm` found
- CPU fallback otherwise

### Force a clean rebuild (e.g. after ROCm update)
```bash
# SSH to node and clear build dir first
ssh johanus@192.168.1.54 "rm -rf ~/llama.cpp/build"
bash scripts/llama_bench.sh bosgame-m5 --install
```

---

## ROCm / Strix Halo Notes

The Strix requires these env vars for GPU recognition. They are set in `/etc/environment` on bosgame-m5:
```
HSA_OVERRIDE_GFX_VERSION=11.5.1
HSA_ENABLE_SDMA=0
GPU_MAX_HEAP_SIZE=100
```

`libxml2.so.2` must be symlinked on Ubuntu 25.x (package renamed to `libxml2-16`):
```bash
sudo apt-get install -y libxml2-16
sudo ln -sf /usr/lib/x86_64-linux-gnu/libxml2.so.16 /usr/lib/x86_64-linux-gnu/libxml2.so.2
sudo ldconfig
```

Ollama on the Strix works fine without any of this — it bundles its own ROCm libraries. The above is only needed for standalone llama-bench.

---

## Downloading GGUFs for llama-bench

Ollama stores models as internal blobs that llama-bench cannot read. For llama-bench, models need proper GGUFs downloaded separately. Use `hf` CLI on the node:

```bash
ssh johanus@192.168.1.54
mkdir -p ~/models/<model-slug>
~/.local/bin/hf download <owner>/<repo> \
  --include '<filename>.gguf' \
  --local-dir ~/models/<model-slug>
```

### Known working HF repos (Q4_K_M)

| Model | HF Repo | Filename |
|-------|---------|----------|
| qwen3.5:27b | bartowski/Qwen_Qwen3.5-27B-GGUF | Qwen_Qwen3.5-27B-Q4_K_M.gguf |
| qwen3.5:35b-a3b | bartowski/Qwen_Qwen3.5-35B-A3B-GGUF | Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf |
| qwen2.5-coder:14b | bartowski/Qwen2.5-Coder-14B-Instruct-GGUF | Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf |
| qwen2.5-coder:32b | bartowski/Qwen2.5-Coder-32B-Instruct-GGUF | Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf |
| qwen3-coder:30b | bartowski/Qwen_Qwen3-30B-A3B-GGUF | Qwen_Qwen3-30B-A3B-Q4_K_M.gguf |
| qwen3.5:122b | bartowski/Qwen_Qwen3.5-122B-A10B-GGUF | Qwen_Qwen3.5-122B-A10B-Q4_K_M/ (2 shards) |

Update `source`, `hf_repo`, and `hf_file` in `config/models.yaml` after downloading so future llama-bench runs find the file automatically.

---

## Monitoring

### GPU hardware metrics (Strix)
```bash
ssh johanus@192.168.1.54
sudo radeontop
```

### Thermals
```bash
# Strix
ssh johanus@192.168.1.54 "sensors"

# GB10
ssh wally@192.168.1.52 "sensors"
```

### Check Ollama is running on a node
```bash
curl http://192.168.1.54:11434/api/tags | python3 -m json.tool | grep name
curl http://192.168.1.52:11434/api/tags | python3 -m json.tool | grep name
```

---

## Preflight Checks

Before running benchmarks, verify connectivity and Ollama health:
```bash
# SSH
ssh -i ~/.ssh/id_ed25519 johanus@192.168.1.54 "echo ok"
ssh -i ~/.ssh/id_ed25519 wally@192.168.1.52 "echo ok"

# Ollama
curl -s http://192.168.1.54:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']), 'models')"
curl -s http://192.168.1.52:11434/api/tags | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']), 'models')"
```

---

## Benchmarking Another Model (Quick Reference)

See [adding-a-model.md](adding-a-model.md) for full details. Short version:

1. Pull via Ollama on the node: `ollama pull <tag>`
2. Download GGUF to `~/models/<slug>/` for llama-bench
3. Add entry to `config/models.yaml` with `hf_repo` and `hf_file`
4. Add to `node_models` section for relevant nodes
5. Run: `./scripts/run_all.sh <node> --models "<tag>"`
