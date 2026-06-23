#!/usr/bin/env python3
"""
Benchmark each model through the gateway.
Reports tokens/sec, latency, and token counts per prompt.

Usage:
    python3 benchmark.py                    # sequential then concurrent
    python3 benchmark.py --gateway http://192.168.0.30:8080
    python3 benchmark.py --concurrent-only  # skip sequential
"""

import argparse
import json
import time
import threading
import urllib.request

GATEWAY = "http://192.168.0.30:8080"

MODELS = [
    {
        "label": "llama3:70b (generate)",
        "payload": {"model": "llama3:70b", "task_type": "generate"},
    },
    {
        "label": "gemma4:12b (classify)",
        "payload": {"model": "gemma4:12b", "task_type": "classify"},
    },
]

PROMPTS = [
    {"name": "short",  "prompt": "What is machine learning? Answer in one sentence."},
    {"name": "medium", "prompt": "Explain how transformers work in neural networks. Be concise."},
    {"name": "long",   "prompt": "Compare supervised, unsupervised, and reinforcement learning. Give an example of each."},
]


def call_gateway(gateway: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{gateway}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def tps(response: dict) -> float:
    eval_count = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 0)
    if eval_duration_ns <= 0 or eval_count <= 0:
        return 0.0
    return eval_count / (eval_duration_ns / 1e9)


def run_sequential(gateway: str) -> None:
    print(f"\n{'='*70}")
    print("SEQUENTIAL (models run one at a time)")
    print(f"{'='*70}")

    for model in MODELS:
        print(f"\n>>> {model['label']}")
        print("-" * 70)

        results = []
        for p in PROMPTS:
            payload = {**model["payload"], "prompt": p["prompt"]}
            print(f"  [{p['name']:6}] ", end="", flush=True)

            wall_start = time.time()
            try:
                resp = call_gateway(gateway, payload)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            wall_sec = time.time() - wall_start

            speed = tps(resp)
            results.append(speed)
            print(
                f"{resp.get('eval_count', 0):4} tokens | {speed:5.1f} tok/s | "
                f"{wall_sec:5.1f}s wall"
            )

        if results:
            print(f"  {'avg':>10}: {sum(results)/len(results):.1f} tok/s   "
                  f"best: {max(results):.1f} tok/s")


def run_concurrent(gateway: str) -> None:
    print(f"\n{'='*70}")
    print("CONCURRENT (both models running simultaneously)")
    print(f"{'='*70}")

    for p in PROMPTS:
        print(f"\n--- prompt: {p['name']} ---")
        results = {}
        errors = {}
        lock = threading.Lock()

        def worker(model: dict) -> None:
            label = model["label"]
            payload = {**model["payload"], "prompt": p["prompt"]}
            wall_start = time.time()
            try:
                resp = call_gateway(gateway, payload)
                wall_sec = time.time() - wall_start
                speed = tps(resp)
                with lock:
                    results[label] = {
                        "tokens": resp.get("eval_count", 0),
                        "tps": speed,
                        "wall": wall_sec,
                    }
            except Exception as e:
                with lock:
                    errors[label] = str(e)

        threads = [threading.Thread(target=worker, args=(m,)) for m in MODELS]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_wall = time.time() - start

        for model in MODELS:
            label = model["label"]
            if label in errors:
                print(f"  {label}: ERROR — {errors[label]}")
            elif label in results:
                r = results[label]
                print(f"  {label}: {r['tokens']:4} tokens | {r['tps']:5.1f} tok/s | {r['wall']:5.1f}s wall")

        print(f"  total wall time: {total_wall:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default=GATEWAY, help="Gateway base URL")
    parser.add_argument("--concurrent-only", action="store_true")
    args = parser.parse_args()

    print(f"Gateway: {args.gateway}")

    if not args.concurrent_only:
        run_sequential(args.gateway)

    run_concurrent(args.gateway)

    print(f"\n{'='*70}")
    print("Done.\n")
