#!/usr/bin/env python3
"""
Benchmark each model through the gateway.
Reports tokens/sec, latency, and token counts per prompt.

Usage:
    python benchmark.py
    python benchmark.py --gateway http://192.168.0.78:8080
"""

import argparse
import json
import time
import urllib.request

GATEWAY = "http://localhost:8080"

MODELS = [
    {
        "label": "llama3:70b (generate)",
        "payload": {"model": "llama3:70b", "task_type": "generate"},
    },
    {
        "label": "gemma4:12b (classify)",
        "payload": {"task_type": "classify"},
    },
]

PROMPTS = [
    {"name": "short", "prompt": "What is machine learning? Answer in one sentence."},
    {"name": "medium", "prompt": "Explain how transformers work in neural networks. Be concise."},
    {"name": "long", "prompt": "Compare supervised, unsupervised, and reinforcement learning. Give an example of each."},
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


def run(gateway: str) -> None:
    print(f"\nGateway: {gateway}")
    print("=" * 70)

    for model in MODELS:
        print(f"\n>>> {model['label']}")
        print("-" * 70)

        results = []
        for p in PROMPTS:
            payload = {**model["payload"], "prompt": p["prompt"]}
            print(f"  [{p['name']}] ", end="", flush=True)

            wall_start = time.time()
            try:
                resp = call_gateway(gateway, payload)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            wall_sec = time.time() - wall_start

            tokens = resp.get("eval_count", 0)
            prompt_tokens = resp.get("prompt_eval_count", 0)
            speed = tps(resp)
            results.append(speed)

            print(
                f"{tokens} tokens | {speed:.1f} tok/s | "
                f"{wall_sec:.1f}s wall | prompt={prompt_tokens} tok"
            )

        if results:
            avg = sum(results) / len(results)
            best = max(results)
            print(f"  {'avg':>8}: {avg:.1f} tok/s   best: {best:.1f} tok/s")

    print("\n" + "=" * 70)
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default=GATEWAY, help="Gateway base URL")
    args = parser.parse_args()
    run(args.gateway)
