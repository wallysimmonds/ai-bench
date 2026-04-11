# GOALS.md — Sovereign AI Benchmarking Program

## Purpose

This program builds the evidence base for sovereign, air-gapped AI deployment in
government and regulated industry. It serves three overlapping work streams:

1. **Immediate client engagement** — a hardware BOM and deployment methodology for a
   fully offline, air-gapped agentic AI and RAG deployment.
2. **Sovereign AI consulting practice** — a repeatable, portable evaluation methodology
   that can be deployed on-site at any client facility.
3. **Fujitsu cloud/RAG work** — baseline benchmarks and quality comparisons that
   inform cloud LLM and RAG architecture decisions.

The benchmark program is not just supporting material for the consulting practice —
it *is* the consulting product. The methodology, tooling, and results produced here
are the deliverable.

---

## Success Criteria

The program succeeds when it can answer the following questions with reproducible,
hardware-specific evidence:

### 1. Performance floor — can the hardware run the model productively?

- Tokens/sec (prefill and generation, per node, per model, per quantisation)
- Time to first token (TTFT) under realistic prompt loads
- Long-context degradation (32K+ prompts, needle-in-haystack retrieval)
- Concurrent request handling (vLLM on nvidia-ai; 2–4 parallel on other nodes)

**Success:** A clear matrix of model × quantisation × node combinations with
acceptable throughput (target: ≥15 tok/s generation for interactive agentic use).

### 2. Quality floor — is the model good enough for the task?

- Instruction following accuracy (IFEval)
- Factual correctness within retrieved context (RAG-specific test sets)
- Tool/function calling reliability (structured output, schema adherence)
- Document retrieval and summarisation quality
- Claude as quality baseline throughout — open models evaluated against it, not
  against each other in isolation

**Success:** Per-model quality scores on sovereign workload task sets, with a clear
"minimum viable quality" threshold identified for the immediate engagement.

### 3. Agentic floor — can the tooling layer close the gap with Claude Code?

- Agentic task completion rate: Aider and OpenCode against sovereign task sets,
  with local models vs Claude as baseline
- Multi-file edit coherence and tool calling reliability
- Context window behaviour under real workload conditions
- Identification of the model × tooling combination that crosses the productive
  agentic use threshold for each workload class

**Success:** A documented crossover point — the minimum model and configuration at
which local agentic tooling is productive enough for a given client workload class.

### 4. Reproducibility — can another operator repeat this?

- All benchmark runs reproducible from `harness/run_benchmark.sh` with a
  `node_config.yaml` as the only per-node input
- Results committed to `results/` in the standard CSV schema
- Node configuration fully documented in `nodes/`

**Success:** A practitioner who has never seen the fleet can deploy the harness on a
new node and produce comparable results within one working session.

---

## Guiding Principles

- **Evidence-based, not consensus-based.** Recommendations derive from benchmarks
  run on real hardware under real workload conditions — not leaderboard aggregation
  or vendor claims.
- **Bare metal for benchmarking.** No Docker overhead noise on inference paths.
  Native Ollama on all nodes.
- **Community-aligned and reproducible.** Benchmark methodology follows community
  standards (llama.cpp, Ollama, vLLM) so results are comparable and credible.
- **Practise what we preach.** Use local models wherever possible throughout the
  program. Use Claude where local models are not yet adequate — and document the gap.
- **Medium-term impulses go to BACKLOG.md.** Current sprint stays focused.

---

## Differentiation

### The question this program answers

Most local LLM benchmarking work measures throughput — tokens/sec, TTFT, quantisation
trade-offs. That work is necessary but not sufficient. The question that matters to
sovereign AI clients is different:

> **What is the minimum viable local model for productive agentic work in a
> sovereign deployment, and how do you configure the tooling layer to get there?**

This program produces evidence-based answers to that question. The benchmark
infrastructure is the means; the consulting product is the answer.

---

### What exists and why it is not enough

The open-source benchmarking landscape is active. Repos such as `ai-bench`,
`awesome-local-llm`, and the Aider leaderboard provide solid throughput and coding
eval baselines. Tools like Aider, OpenCode, Cline, and `ollama-code` have made
local agentic coding genuinely usable.

What is missing:

- **Agentic task completion rate on sovereign workloads.** Almost all published evals
  target SWE-bench, HumanEval, or Aider polyglot — coding benchmarks against open
  GitHub repos. Sovereign clients do not write Next.js apps. Their workloads are
  document processing, RAG over classified corpora, multi-step tool use in isolated
  environments, and instruction following against agency-specific schemas. No
  published benchmark covers this combination at bare metal, air-gapped.

- **Tooling layer evaluation.** The gap between a capable local model and a productive
  agentic workflow is the tooling layer: context window configuration, tool calling
  reliability, multi-file edit coherence, and MCP integration. This is rarely measured
  systematically. In head-to-head testing (OpenCode + Claude Sonnet 4.6 vs Claude Code
  native), the task completion gap was ~9 percentage points — almost entirely explained
  by tooling and model quality differences, not hardware. Understanding that gap on
  open-weight models is the work.

- **Hardware-to-workflow translation.** Fleet operators need to know not just which
  model scores highest, but which model × quantisation × backend × tooling combination
  produces acceptable agentic throughput on their specific hardware. That answer is
  different for a DGX Spark, a Strix Halo APU, and a discrete GPU node.

---

### This program's differentiation

**Evidence base, not consensus.** Recommendations are derived from reproducible
benchmarks run on real hardware under real workload conditions, not from leaderboard
aggregation or vendor claims.

**Sovereign-first workload design.** Test sets are built for the immediate engagement
and generalisable to government and regulated industry use cases — not adapted from
academic benchmarks that assume cloud access and open codebases.

**Full-stack evaluation.** Throughput benchmarks (llama.cpp, Ollama, vLLM) are paired
with agentic task completion tests (Aider, OpenCode against sovereign task sets) and
quality evaluation (Claude as baseline). The result is a three-layer answer: can the
hardware run the model, can the model do the task, can the tooling layer close the gap.

**Portable methodology.** The benchmark harness, node configuration schema, and
evaluation framework are designed for deployment on any node in the fleet and for
reuse in client site visits. The Strix Halo + nvidia-ai combination becomes a portable
evaluation unit that can be brought on-site.

**Practitioner-built.** This program is built by someone deploying sovereign AI for
real clients under real constraints, not a research team optimising for paper
publication. The methodology reflects operational reality.

---

### The Claude Code parity question

Claude Code represents the current ceiling for agentic coding experience. The
open-source equivalents — Aider (terminal-first, Git-native, Ollama-compatible),
OpenCode (Go TUI, 75+ model providers, air-gap ready), and `ollama-code` (forked
from Qwen Code, privacy-first local architecture) — close much of that gap at the
tooling level. The remaining gap is model quality.

This program quantifies that gap on sovereign hardware and identifies the crossover
point: the model and configuration at which local agentic tooling becomes productive
enough for a given client workload class. That crossover point is the core deliverable
of the consulting engagement.
