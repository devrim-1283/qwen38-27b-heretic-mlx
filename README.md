# Qwen3.8-27B-heretic — MLX builds for Apple Silicon

Eight MLX conversions of [`trohrbaugh/Qwen3.8-27B-heretic-ara`](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara),
in two families — text-only and vision-capable — at four precisions each.

The weights live on Hugging Face. This repo holds the conversion recipes, the benchmark
harness, and the numbers behind the model cards.

## The builds

**Text-only** — converted with `mlx-lm`. Smaller, no image input.

| Build | Size | Bits/weight | Generation | Peak memory |
|---|---|---|---|---|
| [4-bit](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-MLX-4bit) | 15.1 GB | 4.501 | **37.9 tok/s** | 15.5 GB |
| [6-bit](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-MLX-6bit) | 21.9 GB | 6.501 | **27.9 tok/s** | 22.2 GB |
| [8-bit](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-MLX-8bit) | 28.6 GB | 8.501 | **22.2 tok/s** | 28.9 GB |
| [bf16](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-MLX-bf16) | 53.8 GB | 16 | **12.7 tok/s** | 54.1 GB |

**Vision** — converted with `mlx-vlm`. Reads images.

| Build | Size | Bits/weight | Generation | Peak memory |
|---|---|---|---|---|
| [4-bit](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-VL-MLX-4bit) | 16.1 GB | 4.695 | **38.9 tok/s** | 19.2 GB |
| [6-bit](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-VL-MLX-6bit) | 22.8 GB | 6.661 | **29.2 tok/s** | 27.0 GB |
| [8-bit](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-VL-MLX-8bit) | 29.5 GB | 8.627 | **23.1 tok/s** | 34.7 GB |
| [bf16](https://huggingface.co/donedynamics/Qwen3.8-27B-heretic-VL-MLX-bf16) | 54.7 GB | 16 | **13.2 tok/s** | 55.8 GB |

All numbers measured on one machine — see [BENCHMARKS.md](BENCHMARKS.md) for the hardware,
the method, and what the numbers do and do not mean.

## Which one

Start with **4-bit**. It is roughly 3x the generation speed of bf16 and needs a third of
the memory, and on ordinary work the quality gap is hard to notice. Move up only when you
have a task where you can measure the difference.

Take the **VL** family if you send images. It costs about 1 GB more on disk and 4-6 GB more
peak memory than its text-only sibling, at the same generation speed. The extra memory is
image-patch activations, not weights.

Take **bf16** when you want a reference point with no quantization error at all — for
judging what the quantized builds lost, or when memory is free and speed is not the
constraint.

## Text-only means text-only

The base model is multimodal. `mlx-lm` converts the language tower only, so the text-only
builds carry no `vision_config` and no vision tensors — all 1847 tensors sit under
`language_model.`. Sending an image gets you nothing. The VL builds keep the encoder: 333
of 2180 tensors under `vision_tower`.

This is easy to get wrong, because both `mlx-lm` and `mlx-vlm` list `qwen3_5` as a
supported architecture. The question is not whether MLX supports the model — it is which
MLX tool you convert with.

## What the base model is

`trohrbaugh/Qwen3.8-27B-heretic-ara` is an **abliterated** derivative of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B): its refusal behaviour has
been surgically removed, so it answers prompts a safety-tuned model would decline.

These conversions change format and precision only. They add no alignment and remove none.
Evaluate before putting one in front of users, and apply your own filtering where your use
case needs it.

Lineage: `Qwen/Qwen3.8-27B` → `trohrbaugh/Qwen3.8-27B-heretic-ara` → these builds.

## Reproduce

```bash
# text-only
pip install mlx-lm
mlx_lm.convert --hf-path trohrbaugh/Qwen3.8-27B-heretic-ara -q --q-bits 4 \
  --mlx-path Qwen3.8-27B-heretic-MLX-4bit

# vision — different tool, not a different flag
pip install mlx-vlm
mlx_vlm.convert --hf-path trohrbaugh/Qwen3.8-27B-heretic-ara -q --q-bits 4 \
  --mlx-path Qwen3.8-27B-heretic-VL-MLX-4bit
```

Drop `-q --q-bits N` for the bf16 build. [`scripts/convert_all.sh`](scripts/convert_all.sh)
runs all eight.

Source revision: `a67ae100d933c0d17af3232bda35825979fc63ce`. Verified before conversion —
7 shards, 1199 tensors, every safetensors header parsed, no missing files.
[`scripts/verify_model.py`](scripts/verify_model.py) does that check.

## Run

```bash
# text
mlx_lm.generate --model donedynamics/Qwen3.8-27B-heretic-MLX-4bit \
  --prompt "Introduce yourself briefly." --max-tokens 256

# vision
mlx_vlm.generate --model donedynamics/Qwen3.8-27B-heretic-VL-MLX-4bit \
  --image photo.png --prompt "What does this image show?" --max-tokens 256
```

### Reasoning mode eats your token budget

The chat template supports `enable_thinking` and `reasoning_effort`. Thinking is on by
default and runs before the answer starts, so a small `max_tokens` can return reasoning
and nothing else. Pass `enable_thinking=False` to `apply_chat_template` for direct answers.

## Serving

`mlx_lm.server` and `mlx_vlm.server` both speak the OpenAI API. Two things worth knowing
before you put either behind a gateway:

- **`mlx_lm.server` has no authentication.** It binds `127.0.0.1` by default; keep it there
  and put auth in front. `mlx_vlm.server` does have `--api-key`.
- **A single `seed` in a request drops the whole batch.** The server treats seeded requests
  as unbatchable and drains the active batch to run it alone. Measured: four concurrent
  users at 116 tok/s aggregate fall to 38 tok/s when one sends a seed. Strip `seed` at the
  gateway.

Concurrency and prompt-cache flags matter more than any environment variable —
[BENCHMARKS.md](BENCHMARKS.md#concurrency) has the measured curve and the settings that
produced it.

## License

Apache-2.0, inherited from the base model. Credit for the model goes to the Qwen team at
Alibaba Group's Tongyi Lab, and for the abliteration to `trohrbaugh`. This repo contributes
the MLX conversions and the measurements.
