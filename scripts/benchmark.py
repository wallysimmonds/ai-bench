#!/usr/bin/env python3
"""
benchmark.py — Core benchmark runner for ai-bench fleet

Metrics captured:
  - tg (token generation): tokens/sec during generation phase
  - pp (prompt processing): tokens/sec during prefill phase
  - TTFT: time to first token (ms)

For standardised pp512/tg128 numbers use scripts/llama_bench.sh instead.

Usage:
    python scripts/benchmark.py --node lenovo-gb10 --suite standard
    python scripts/benchmark.py --node bosgame-m5 --suite coding
    python scripts/benchmark.py --all --suite standard
    python scripts/benchmark.py --node lenovo-gb10 --model qwen3.5:27b
"""

import argparse
import json
import time
import threading
import queue
import requests
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Per-model timeout overrides ───────────────────────────────────────────────
# Large models are slow — give them more time per test.
# Format: substring match → timeout in seconds
MODEL_TIMEOUTS = {
    "122b":  600,
    "120b":  600,
    "80b":   480,
    "72b":   480,
    "70b":   480,
    "35b":   240,
    "32b":   240,
    "27b":   360,
    "26b":   240,
    "14b":   180,
}

def get_model_timeout(model_name, default=120):
    for key, val in MODEL_TIMEOUTS.items():
        if key in model_name:
            return val
    return default

# ── Thinking model detection ──────────────────────────────────────────────────
# These models run extended chain-of-thought by default.
# think:false disables it for fair throughput comparison.
THINKING_MODELS = {
    "qwen3.5",
    "qwen3-coder-next",
    "qwen3-coder",
    "deepseek-r1",
    "gemma4",
}

def is_thinking_model(model_name):
    return any(t in model_name.lower() for t in THINKING_MODELS)

# ── Benchmark suites ──────────────────────────────────────────────────────────

SUITES = {
    "standard": [
        {
            "id": "pp_512",
            "name": "Prefill — 512 token prompt",
            "prompt": (
                "You are a senior infrastructure architect. Your task is to write a "
                "detailed technical design document. Consider the following requirements carefully "
                "before responding. The system must handle 10,000 concurrent users, provide "
                "sub-100ms latency at the 99th percentile, support multi-region active-active "
                "deployment across at least three geographic regions, implement zero-downtime "
                "deployments with automated rollback capabilities, maintain 99.99% availability "
                "SLA, support horizontal scaling with auto-scaling policies, implement "
                "comprehensive observability including distributed tracing, metrics aggregation, "
                "and centralised logging, provide end-to-end encryption for all data in transit "
                "and at rest, support RBAC with fine-grained permissions, integrate with existing "
                "enterprise identity providers via SAML 2.0 and OIDC, implement rate limiting and "
                "DDoS protection at the edge, support blue-green and canary deployment strategies, "
                "maintain full audit trails for compliance, and implement chaos engineering "
                "practices. Given all these requirements, describe the high-level architecture."
            ),
            "num_predict": 128,
            "measure": "pp",
        },
        {
            "id": "tg_128",
            "name": "Generation — 128 tokens",
            "prompt": "Write a Python function to perform binary search on a sorted list.",
            "num_predict": 128,
            "measure": "tg",
        },
        {
            "id": "tg_500",
            "name": "Generation — 500 tokens",
            "prompt": "Write a Python function to perform binary search on a sorted list.",
            "num_predict": 500,
            "measure": "tg",
        },
        {
            "id": "ttft_short",
            "name": "TTFT — Short prompt",
            "prompt": "What is 2 + 2?",
            "measure": "ttft",
        },
        {
            "id": "ttft_long",
            "name": "TTFT — Long context prompt",
            "prompt": (
                "You are a senior infrastructure architect. Explain in detail the differences "
                "between Azure ExpressRoute and VPN Gateway, covering bandwidth, latency, "
                "reliability, cost model, and typical enterprise use cases. Be thorough."
            ),
            "measure": "ttft",
        },
        {
            "id": "throughput_medium",
            "name": "Throughput — Medium generation",
            "prompt": (
                "Write a Python FastAPI endpoint that accepts a JSON body with fields: "
                "query (string), top_k (int, default 5), and returns a list of document "
                "chunks from a vector store. Include proper error handling, Pydantic models, "
                "and comments."
            ),
            "measure": "throughput",
        },
        {
            "id": "throughput_long",
            "name": "Throughput — Long generation",
            "prompt": (
                "Write a complete Bicep template to deploy an Azure Container App with: "
                "a container registry, managed identity, key vault reference for secrets, "
                "a storage account with blob container, and all necessary role assignments. "
                "Include parameters for environment (dev/prod) and resource naming conventions."
            ),
            "measure": "throughput",
        },
    ],
    "coding": [
        {
            "id": "code_simple",
            "name": "Coding — Simple function",
            "prompt": (
                "Write a Python function that takes a list of dictionaries with keys "
                "'title', 'content', 'embedding' and returns the top_k most similar to "
                "a query embedding using cosine similarity. Include type hints."
            ),
            "measure": "quality",
        },
        {
            "id": "code_multifile",
            "name": "Coding — Multi-file awareness",
            "prompt": (
                "Given this FastAPI app structure:\n"
                "- main.py: FastAPI app with /search endpoint\n"
                "- models.py: Pydantic models for SearchRequest and SearchResult\n"
                "- vector_store.py: class VectorStore with method search(query_embedding, top_k)\n"
                "- embeddings.py: function get_embedding(text) -> list[float]\n\n"
                "Write the complete implementation of vector_store.py using in-memory numpy arrays. "
                "The search method should return SearchResult objects sorted by cosine similarity."
            ),
            "measure": "quality",
        },
        {
            "id": "code_debug",
            "name": "Coding — Debug and fix",
            "prompt": (
                "Fix the bug in this Python code:\n\n"
                "def cosine_similarity(a, b):\n"
                "    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))\n\n"
                "def search(query_embedding, embeddings, top_k=5):\n"
                "    scores = [cosine_similarity(query_embedding, e) for e in embeddings]\n"
                "    top_indices = np.argsort(scores)[:top_k]  # bug here\n"
                "    return [(i, scores[i]) for i in top_indices]\n\n"
                "Explain the bug, fix it, and add a docstring."
            ),
            "measure": "quality",
        },
        {
            "id": "code_infra",
            "name": "Coding — Infrastructure task",
            "prompt": (
                "Write a Python script that uses the Azure SDK to list all Container Apps "
                "in a subscription, check if each one has a managed identity assigned, and "
                "output a CSV with columns: resource_group, app_name, has_identity, "
                "identity_type, last_modified."
            ),
            "measure": "quality",
        },
    ],
    "deployment": [
        {
            "id": "deploy_rag_prompt",
            "name": "Deployment — RAG solution description",
            "prompt": (
                "You are an AI assistant with terminal access. Deploy a minimal RAG "
                "(Retrieval Augmented Generation) solution with these components:\n"
                "1. A FastAPI backend with /upload and /query endpoints\n"
                "2. A simple in-memory vector store\n"
                "3. Ollama as the LLM backend (assume it's running on localhost:11434)\n"
                "4. A React frontend with a file upload and chat interface\n\n"
                "List exactly what files you would create and the steps to deploy this. "
                "Be specific about file paths and commands."
            ),
            "measure": "quality",
            "notes": "Manual evaluation — compare against Claude Code baseline",
        },
    ],
}

# ── Core benchmark functions ─────────────────────────────────────────────────

def load_config():
    with open(CONFIG_DIR / "nodes.yaml") as f:
        nodes = yaml.safe_load(f)
    with open(CONFIG_DIR / "models.yaml") as f:
        models = yaml.safe_load(f)
    return nodes["nodes"], models


def get_models_for_node(node_name, models_config):
    return models_config.get("node_models", {}).get(node_name, [])


def run_ollama_benchmark(host, port, model, prompt, timeout=120, num_predict=None):
    """Run a single benchmark against an Ollama endpoint.

    Captures both tg (generation) and pp (prefill) tokens/sec from Ollama's
    done chunk. timeout is enforced as a wall-clock limit via a daemon thread.
    Thinking models have CoT disabled via the API for fair comparison.

    Args:
        num_predict: if set, caps generation length (used for fixed tg128/pp512 tests)
    """
    url = f"http://{host}:{port}/api/generate"
    thinking = is_thinking_model(model)

    options = {"temperature": 0, "seed": 42}
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": options,
    }
    if thinking:
        payload["think"] = False

    result = {
        "model": model,
        "thinking_suppressed": thinking,
        "prompt_tokens": None,
        "eval_tokens": None,
        "ttft_ms": None,
        "total_ms": None,
        "tg_tokens_per_sec": None,   # generation speed (eval phase)
        "pp_tokens_per_sec": None,   # prefill speed (prompt processing phase)
        "response_preview": "",
        "error": None,
    }

    def _run():
        try:
            t_start = time.time()
            first_token_time = None
            full_response = []

            with requests.post(url, json=payload, stream=True, timeout=(10, timeout)) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if first_token_time is None and chunk.get("response"):
                        first_token_time = time.time()
                        result["ttft_ms"] = round((first_token_time - t_start) * 1000)

                    if chunk.get("response"):
                        full_response.append(chunk["response"])

                    if chunk.get("done"):
                        t_end = time.time()
                        result["total_ms"] = round((t_end - t_start) * 1000)
                        result["prompt_tokens"] = chunk.get("prompt_eval_count")
                        result["eval_tokens"] = chunk.get("eval_count")

                        # tg: generation speed from Ollama's eval_duration
                        if chunk.get("eval_duration") and chunk.get("eval_count"):
                            result["tg_tokens_per_sec"] = round(
                                chunk["eval_count"] / (chunk["eval_duration"] / 1e9), 1
                            )

                        # pp: prefill speed from prompt_eval_duration
                        if chunk.get("prompt_eval_duration") and chunk.get("prompt_eval_count"):
                            result["pp_tokens_per_sec"] = round(
                                chunk["prompt_eval_count"] / (chunk["prompt_eval_duration"] / 1e9), 1
                            )
                        break

            result["response_preview"] = "".join(full_response)[:300]

        except requests.exceptions.ConnectionError:
            result["error"] = f"Connection refused — is Ollama running on {host}:{port}?"
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        result["error"] = f"Timeout after {timeout}s"

    return result


def check_ollama_health(host, port):
    try:
        r = requests.get(f"http://{host}:{port}/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []


def run_node_benchmark(node_name, node_config, models_to_test, suite_name, suite):
    host = node_config["host"]
    port = node_config.get("ollama_port", 11434)

    print(f"\n{'='*60}")
    print(f"Node: {node_name} ({host}:{port})")
    print(f"Suite: {suite_name}")
    print(f"{'='*60}")

    healthy, available_models = check_ollama_health(host, port)
    if not healthy:
        print(f"  ✗ Ollama not reachable at {host}:{port}")
        return []

    print(f"  ✓ Ollama online — {len(available_models)} models loaded")

    run_results = []

    for model in models_to_test:
        model_available = any(model in m for m in available_models)
        if not model_available:
            print(f"\n  Model: {model} — NOT AVAILABLE (run setup script)")
            continue

        timeout = get_model_timeout(model)
        print(f"\n  Model: {model} (timeout: {timeout}s/test)")

        # Warm up — ensure the model is loaded before timing starts.
        # Large models can take 60-120s to load.
        print(f"    [warm-up] Loading model...", end=" ", flush=True)
        warmup = run_ollama_benchmark(host, port, model, "hi", timeout=300)
        if warmup.get("error"):
            print(f"FAILED ({warmup['error']}) — skipping")
            continue
        print("ready")

        for test in suite:
            print(f"    [{test['id']}] {test['name']}...", end=" ", flush=True)
            result = run_ollama_benchmark(
                host, port, model, test["prompt"],
                timeout=timeout,
                num_predict=test.get("num_predict"),
            )

            record = {
                "timestamp": datetime.now().isoformat(),
                "node": node_name,
                "node_type": node_config.get("type", "unknown"),
                "model": model,
                "test_id": test["id"],
                "test_name": test["name"],
                "suite": suite_name,
                **result,
            }
            run_results.append(record)

            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                tg = f"tg {result['tg_tokens_per_sec']} tok/s" if result["tg_tokens_per_sec"] else ""
                pp = f"pp {result['pp_tokens_per_sec']} tok/s" if result["pp_tokens_per_sec"] else ""
                ttft = f"TTFT {result['ttft_ms']}ms" if result["ttft_ms"] else ""
                print(f"✓  {tg}  {pp}  {ttft}".strip())

    return run_results


def save_results(results, node, suite):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"{ts}_{node}_{suite}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {filename}")
    return filename


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AI Fleet Benchmark Runner")
    parser.add_argument("--node", help="Node name from config/nodes.yaml")
    parser.add_argument("--all", action="store_true", help="Run on all nodes")
    parser.add_argument("--suite", default="standard",
                        choices=list(SUITES.keys()), help="Benchmark suite")
    parser.add_argument("--model", help="Test a specific model only")
    parser.add_argument("--list-nodes", action="store_true")
    args = parser.parse_args()

    nodes, models_config = load_config()

    if args.list_nodes:
        print("Configured nodes:")
        for name, cfg in nodes.items():
            print(f"  {name}: {cfg['host']} ({cfg.get('type','?')})")
        return

    suite = SUITES[args.suite]
    target_nodes = list(nodes.keys()) if args.all else [args.node]

    if not args.all and not args.node:
        parser.print_help()
        sys.exit(1)

    all_results = []

    for node_name in target_nodes:
        if node_name not in nodes:
            print(f"Unknown node: {node_name}")
            continue

        node_config = nodes[node_name]

        if args.model:
            models_to_test = [args.model]
        else:
            models_to_test = get_models_for_node(node_name, models_config)

        results = run_node_benchmark(
            node_name, node_config, models_to_test, args.suite, suite
        )
        all_results.extend(results)

        if results:
            save_results(results, node_name, args.suite)

    print(f"\n{'='*60}")
    print(f"Complete. {len(all_results)} benchmark records collected.")
    print(f"Run: python scripts/report_excel.py --results results/")
    print(f"     python scripts/report_html.py --results results/")


if __name__ == "__main__":
    main()
