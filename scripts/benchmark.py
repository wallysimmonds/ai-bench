#!/usr/bin/env python3
"""
benchmark.py — Core benchmark runner for ai-bench fleet

Usage:
    python scripts/benchmark.py --node nvidia-ai --suite standard
    python scripts/benchmark.py --node bosgame-m5 --suite coding
    python scripts/benchmark.py --all --suite standard
    python scripts/benchmark.py --node lenovo-gb10 --model qwen3.5:27b
"""

import argparse
import json
import time
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

# ── Benchmark prompts ────────────────────────────────────────────────────────

SUITES = {
    "standard": [
        {
            "id": "ttft_short",
            "name": "TTFT — Short prompt",
            "prompt": "What is 2 + 2?",
            "measure": "ttft",
        },
        {
            "id": "ttft_long",
            "name": "TTFT — Long context prompt",
            "prompt": "You are a senior infrastructure architect. Explain in detail the differences between Azure ExpressRoute and VPN Gateway, covering bandwidth, latency, reliability, cost model, and typical enterprise use cases. Be thorough.",
            "measure": "ttft",
        },
        {
            "id": "throughput_medium",
            "name": "Throughput — Medium generation",
            "prompt": "Write a Python FastAPI endpoint that accepts a JSON body with fields: query (string), top_k (int, default 5), and returns a list of document chunks from a vector store. Include proper error handling, Pydantic models, and comments.",
            "measure": "throughput",
        },
        {
            "id": "throughput_long",
            "name": "Throughput — Long generation",
            "prompt": "Write a complete Bicep template to deploy an Azure Container App with: a container registry, managed identity, key vault reference for secrets, a storage account with blob container, and all necessary role assignments. Include parameters for environment (dev/prod) and resource naming conventions.",
            "measure": "throughput",
        },
    ],
    "coding": [
        {
            "id": "code_simple",
            "name": "Coding — Simple function",
            "prompt": "Write a Python function that takes a list of dictionaries with keys 'title', 'content', 'embedding' and returns the top_k most similar to a query embedding using cosine similarity. Include type hints.",
            "measure": "quality",
        },
        {
            "id": "code_multifile",
            "name": "Coding — Multi-file awareness",
            "prompt": """Given this FastAPI app structure:
- main.py: FastAPI app with /search endpoint
- models.py: Pydantic models for SearchRequest and SearchResult
- vector_store.py: class VectorStore with method search(query_embedding, top_k)
- embeddings.py: function get_embedding(text) -> list[float]

Write the complete implementation of vector_store.py using in-memory numpy arrays. The search method should return SearchResult objects sorted by cosine similarity.""",
            "measure": "quality",
        },
        {
            "id": "code_debug",
            "name": "Coding — Debug and fix",
            "prompt": """Fix the bug in this Python code:

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_embedding, embeddings, top_k=5):
    scores = [cosine_similarity(query_embedding, e) for e in embeddings]
    top_indices = np.argsort(scores)[:top_k]  # bug here
    return [(i, scores[i]) for i in top_indices]

Explain the bug, fix it, and add a docstring.""",
            "measure": "quality",
        },
        {
            "id": "code_infra",
            "name": "Coding — Infrastructure task",
            "prompt": "Write a Python script that uses the Azure SDK to list all Container Apps in a subscription, check if each one has a managed identity assigned, and output a CSV with columns: resource_group, app_name, has_identity, identity_type, last_modified.",
            "measure": "quality",
        },
    ],
    "deployment": [
        {
            "id": "deploy_rag_prompt",
            "name": "Deployment — RAG solution description",
            "prompt": """You are an AI assistant with terminal access. Deploy a minimal RAG (Retrieval Augmented Generation) solution with these components:
1. A FastAPI backend with /upload and /query endpoints
2. A simple in-memory vector store
3. Ollama as the LLM backend (assume it's running on localhost:11434)
4. A React frontend with a file upload and chat interface

List exactly what files you would create and the steps to deploy this. Be specific about file paths and commands.""",
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
    node_models = models_config.get("node_models", {}).get(node_name, [])
    return node_models

def run_ollama_benchmark(host, port, model, prompt, timeout=120):
    """Run a single benchmark against an Ollama endpoint."""
    url = f"http://{host}:{port}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,   # deterministic for benchmarking
            "seed": 42,
        }
    }

    result = {
        "model": model,
        "prompt_tokens": None,
        "eval_tokens": None,
        "ttft_ms": None,
        "total_ms": None,
        "tokens_per_sec": None,
        "response_preview": "",
        "error": None,
    }

    try:
        t_start = time.time()
        first_token_time = None
        full_response = []

        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
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
                    if chunk.get("eval_duration") and chunk.get("eval_count"):
                        result["tokens_per_sec"] = round(
                            chunk["eval_count"] / (chunk["eval_duration"] / 1e9), 1
                        )
                    break

        result["response_preview"] = "".join(full_response)[:300]

    except requests.exceptions.ConnectionError:
        result["error"] = f"Connection refused — is Ollama running on {host}:{port}?"
    except requests.exceptions.Timeout:
        result["error"] = f"Timeout after {timeout}s"
    except Exception as e:
        result["error"] = str(e)

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
        # Check if model is actually available
        model_available = any(model in m for m in available_models)
        if not model_available:
            print(f"\n  Model: {model} — NOT AVAILABLE (run setup script)")
            continue

        print(f"\n  Model: {model}")

        for test in suite:
            print(f"    [{test['id']}] {test['name']}...", end=" ", flush=True)
            result = run_ollama_benchmark(host, port, model, test["prompt"])

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
                tps = f"{result['tokens_per_sec']} tok/s" if result["tokens_per_sec"] else "N/A"
                ttft = f"TTFT {result['ttft_ms']}ms" if result["ttft_ms"] else ""
                print(f"✓  {tps}  {ttft}")

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
