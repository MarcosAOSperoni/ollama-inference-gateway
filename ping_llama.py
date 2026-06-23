#!/usr/bin/env python3
"""
Quick ping test for inference models through the gateway.
Sends a single short prompt and reports tok/s and latency.

Usage:
    python3 ping_llama.py
    python3 ping_llama.py --model gemma4:12b
    python3 ping_llama.py --all
    python3 ping_llama.py --gateway http://192.168.0.30:8080 --all
"""

import argparse
import json
import time
import urllib.request

GATEWAY = "http://192.168.0.30:8080"
PROMPT = "Reply with one word: hello."
MODELS = ["llama3:70b", "gemma4:12b"]


def call_gateway(gateway: str, model: str) -> dict:
    payload = {"model": model, "prompt": PROMPT}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{gateway}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def ping(gateway: str, model: str) -> None:
    print(f"Pinging {model} via {gateway}...")
    wall_start = time.time()
    try:
        resp = call_gateway(gateway, model)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return
    wall_sec = time.time() - wall_start

    eval_count = resp.get("eval_count", 0)
    eval_duration_ns = resp.get("eval_duration", 0)
    tps = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0.0

    print(f"  response   : {resp.get('response', '').strip()}")
    print(f"  tokens     : {eval_count}")
    print(f"  tok/s      : {tps:.1f}")
    print(f"  wall time  : {wall_sec:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway", default=GATEWAY)
    parser.add_argument("--model", default="llama3:70b")
    parser.add_argument("--all", action="store_true", help="Ping all models")
    args = parser.parse_args()

    models = MODELS if args.all else [args.model]
    for model in models:
        ping(args.gateway, model)
