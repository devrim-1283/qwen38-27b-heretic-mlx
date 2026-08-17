# Benchmarks

Every number here came off one machine. Treat them as an order-of-magnitude guide for
similar hardware, not as a benchmark suite.

## Hardware and software

| | |
|---|---|
| Machine | Mac Studio, Apple M3 Ultra |
| Memory | 512 GB unified |
| GPU memory ceiling | `iogpu.wired_limit_mb = 491520` (480 GiB) |
| OS | macOS 26.5.2 |
| Text builds | `mlx` 0.32.0, `mlx-lm` 0.31.3 |
| Vision builds | `mlx-vlm` 0.6.13, `transformers` 5.15.0 |
| Python | 3.12.14 |

## Single-stream generation

One request at a time, greedy decoding, via `mlx_lm.generate` / `mlx_vlm.generate`.
Text prompts were 68 tokens for 120-150 generated tokens; vision prompts were 470 tokens
(image + text) for 120 generated tokens.

| Build | Family | Generation | Peak memory |
|---|---|---|---|
| 4-bit | text | 37.9 tok/s | 15.5 GB |
| 6-bit | text | 27.9 tok/s | 22.2 GB |
| 8-bit | text | 22.2 tok/s | 28.9 GB |
| bf16 | text | 12.7 tok/s | 54.1 GB |
| 4-bit | vision | 38.9 tok/s | 19.2 GB |
| 6-bit | vision | 29.2 tok/s | 27.0 GB |
| 8-bit | vision | 23.1 tok/s | 34.7 GB |
| bf16 | vision | 13.2 tok/s | 55.8 GB |

For reference, the same bf16 weights under `transformers` on the MPS backend ran at
8.7 tok/s — the MLX bf16 build is about 45% faster at identical precision.

### Prompt processing barely moves

Prefill ran at 303-331 tok/s across all four vision builds and 74-153 tok/s across the text
builds depending on prompt length. Quantization level has almost no effect on prefill; the
whole spread between builds shows up in generation. If your workload is long prompts and
short answers, the gap between 4-bit and bf16 is much smaller than the table suggests.

### Vision costs memory, not speed

Each vision build generates at the same rate as its text-only sibling — slightly faster, in
fact. It carries about 1 GB more weights but 4-6 GB more peak memory. That gap is
image-patch activations, not parameters. Pick the text-only build to save memory, not to go
faster.

## Concurrency

`mlx_lm.server` batches concurrent requests rather than serialising them. Measured on the
text 4-bit build, 120 tokens per request, non-streaming, over HTTP.

With `--decode-concurrency 16 --prompt-concurrency 8`:

| Concurrent requests | Aggregate | Per request | Wall time |
|---|---|---|---|
| 1 | 28.4 tok/s | 28.4 tok/s | 4.2 s |
| 4 | 73.8 tok/s | 18.5 tok/s | 6.5 s |
| 8 | **102.3 tok/s** | 12.9 tok/s | 9.4 s |
| 12 | 93.5 tok/s | 7.8 tok/s | 15.4 s |

Aggregate throughput peaks around 8 concurrent requests at roughly 3.6x the single-stream
rate, then declines. Per-request speed falls the whole way — throughput and latency pull in
opposite directions, so the right setting depends on whether you are optimising for one
user waiting or many users served.

Single-stream over HTTP (28.4 tok/s) is lower than the 37.9 tok/s measured through
`mlx_lm.generate`, which is the cost of the HTTP path and non-streaming response assembly.

### Raise both concurrency flags together

An earlier run with `--decode-concurrency 8 --prompt-concurrency 4` collapsed at 8
concurrent requests — aggregate throughput fell from 85 to 50 tok/s. The bottleneck was not
decode but the prefill queue: eight prompts through a four-wide prefill stage forced two
serialised rounds. Raising only the decode width makes the system quietly worse under load.

| Concurrent | prompt-concurrency 4 | prompt-concurrency 8 |
|---|---|---|
| 4 | 85.1 tok/s | 73.8 tok/s |
| 8 | 50.1 tok/s ❌ | 102.3 tok/s |

## KV cache is cheap on this architecture

The model runs hybrid attention — 64 layers, of which 48 are `linear_attention` and 16 are
`full_attention` (`full_attention_interval: 4`). Only the 16 full-attention layers keep a
KV cache that grows with sequence length; the rest hold a constant-size recurrent state.

With `num_key_value_heads: 4` and `head_dim: 256`, each token costs
`2 x 4 x 256 x 2 bytes x 16 layers` = 64 KB.

| Context | KV cache |
|---|---|
| 32K tokens | 2.0 GB |
| 128K tokens | 8.0 GB |
| 262K (max) | 16.0 GB |

A conventional 64-layer model of this size would want roughly four times that. Long
contexts are not what limits how many models you can hold resident — weights are.

There is a floor to be aware of: the 48 linear-attention layers carry a recurrent state of
about 154 MB per concurrent sequence, independent of context length. Five concurrent users
start at 0.8 GB before a single token of KV cache.

## Things that did not help

`MLX_METAL_FAST_SYNCH` and `MLX_MAX_OPS_PER_BUFFER` were A/B tested at batch 1 and batch 4.
The spread was under 1% (37.3 / 37.3 / 37.2 / 36.8 tok/s at batch 1). These are frequently
recommended online; on this workload they do nothing.

Raising `iogpu.wired_limit_mb` was also a no-op here — it was already at 480 GiB, and
`mlx_lm.server` calls `mx.set_wired_limit` itself at startup. It is a ceiling, not a
reservation.

## Method

Single runs, one machine, one prompt shape per measurement. No warm-up discarding, no
repetition, no confidence intervals. Page cache state was not controlled between runs,
which matters for load time but not for generation rate.

Model load time in particular should be ignored: converted models loaded in about 6 seconds
because their files had just been written and were still in the page cache. A cold load
from disk is substantially slower.

Reproduce with [`scripts/bench_concurrency.py`](scripts/bench_concurrency.py).
