#!/usr/bin/env python3
"""Measure how an MLX server scales with concurrent requests.

Start the server first, then point this at it:

    mlx_lm.server --model ~/models/Qwen3.8-27B-heretic-MLX-4bit \
      --host 127.0.0.1 --port 8090 \
      --decode-concurrency 16 --prompt-concurrency 8 --prompt-cache-size 16

    python bench_concurrency.py --port 8090 \
      --model ~/models/Qwen3.8-27B-heretic-MLX-4bit

Raise --decode-concurrency and --prompt-concurrency together. Leaving prefill
narrow while widening decode makes throughput fall under load instead of rise:
prompts queue in serialised rounds and every request waits.

Reports aggregate throughput (how much the machine produces) alongside
per-request throughput (how fast one user feels it). They move in opposite
directions, and which one matters depends on who you are serving.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Explain the advantages of a microservice architecture in three points.",
    "What is the difference between a list and a tuple in Python?",
    "What are the core differences between TCP and UDP?",
    "Why does a database index speed up queries?",
    "What is the difference between a Docker image and a container?",
    "Compare REST and GraphQL briefly.",
    "What is the difference between git rebase and git merge?",
    "Why is Redis fast? Explain briefly.",
]


def one_request(url, model, prompt, max_tokens, results, lock):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            data = json.loads(r.read())
        produced = data.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        print(f"    request failed: {e}", file=sys.stderr)
        produced = 0
    with lock:
        results.append((time.time() - t0, produced))


def run(url, model, n, max_tokens):
    results, lock = [], threading.Lock()
    threads = [
        threading.Thread(target=one_request,
                         args=(url, model, PROMPTS[i % len(PROMPTS)],
                               max_tokens, results, lock))
        for i in range(n)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    total = sum(r[1] for r in results)
    lats = sorted(r[0] for r in results)
    mean_lat = statistics.mean(lats)
    return dict(
        n=n, wall=wall, total=total,
        aggregate=total / wall if wall else 0,
        per_request=(total / len(results)) / mean_lat if results and mean_lat else 0,
        lat_min=lats[0], lat_max=lats[-1],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model", required=True, help="model id the server reports at /v1/models")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--levels", default="1,2,4,8,12",
                    help="comma-separated concurrency levels")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    levels = [int(x) for x in args.levels.split(",")]

    print("%5s %9s %8s %14s %14s %10s %10s" %
          ("N", "wall", "tokens", "aggregate t/s", "per-req t/s", "lat min", "lat max"))
    for n in levels:
        r = run(url, args.model, n, args.max_tokens)
        print("%5d %8.1fs %8d %14.1f %14.1f %9.1fs %9.1fs" %
              (r["n"], r["wall"], r["total"], r["aggregate"],
               r["per_request"], r["lat_min"], r["lat_max"]))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
