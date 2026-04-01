# ai-bench

Local AI infrastructure benchmarking suite — performance testing and agentic deployment evaluation across a heterogeneous inference fleet.

## Fleet

| Node | Hardware | Memory | Notes |
|------|----------|--------|-------|
| `nvidia-ai` | RTX 5070 Ti 16GB + RTX PRO 4000 Blackwell 24GB | ~40GB VRAM | x86, dual GPU |
| `bosgame-m5` | AMD AI Max 395 (Strix Halo) | 128GB DDR5 unified | Linux Mint |
| `lenovo-gb10` | Grace Blackwell GB10 | TBD | DGX Spark clone, ARM |

## Goals

1. **Hardware benchmark matrix** — tokens/sec, TTFT, sustained throughput per node per model
2. **Agentic deployment testing** — can OpenClaw/NemoClaw deploy a RAG solution with minimal prompting vs Claude Code baseline
3. **Infrastructure automation limits** — where does AI-driven infra management break with noname switches / mixed environments

## Structure

```
ai-bench/
├── config/
│   ├── nodes.yaml          # Node definitions and SSH config
│   └── models.yaml         # Model list per node
├── scripts/
│   ├── setup/              # Per-node Ollama + model pull scripts
│   ├── benchmark.py        # Core benchmark runner
│   ├── report_excel.py     # Excel report generator
│   └── report_html.py      # HTML report generator
├── deployments/
│   └── rag-poc/            # RAG deployment test (baseline: Claude Code)
├── results/                # Raw JSON results per run
├── reports/                # Generated Excel/HTML reports
└── docs/
    └── sessions.md         # Session log — goal, result, delta
```

## Quick Start

```bash
# 1. Configure your nodes
cp config/nodes.yaml.example config/nodes.yaml
# edit nodes.yaml with your IPs and SSH keys

# 2. Run setup on a node (pulls Ollama + models)
./scripts/setup/setup-nvidia.sh
./scripts/setup/setup-strix.sh
./scripts/setup/setup-gb10.sh

# 3. Run benchmarks
python scripts/benchmark.py --node nvidia-ai --suite standard
python scripts/benchmark.py --node bosgame-m5 --suite standard
python scripts/benchmark.py --node lenovo-gb10 --suite standard

# 4. Generate report
python scripts/report_excel.py --results results/
python scripts/report_html.py --results results/
```

## Benchmark Suites

- `standard` — throughput, TTFT, sustained agentic session
- `coding` — fixed coding tasks against rag-poc codebase (quality delta)
- `deployment` — agentic RAG deployment attempt (pass/fail + intervention count)

## Session Log

See `docs/sessions.md` — each session has a defined goal and documented result.
