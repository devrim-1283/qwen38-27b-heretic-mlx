#!/usr/bin/env bash
# Convert Qwen3.8-27B-heretic-ara to all eight MLX builds.
#
# mlx-lm and mlx-vlm pin different dependency versions, so they go in separate
# environments. Installing both in one venv is how you end up with a text-only
# "vision" build.
set -euo pipefail

SRC="trohrbaugh/Qwen3.8-27B-heretic-ara"
OUT="${OUT:-$HOME/models}"

LM="${LM_VENV:-$HOME/venvs/qwen}"
VLM="${VLM_VENV:-$HOME/venvs/vlm}"

mkdir -p "$OUT"

echo "==> text-only builds (mlx-lm)"
for bits in 4 6 8; do
  echo "--- ${bits}-bit"
  "$LM/bin/mlx_lm.convert" --hf-path "$SRC" -q --q-bits "$bits" \
    --mlx-path "$OUT/Qwen3.8-27B-heretic-MLX-${bits}bit"
done
echo "--- bf16"
"$LM/bin/mlx_lm.convert" --hf-path "$SRC" \
  --mlx-path "$OUT/Qwen3.8-27B-heretic-MLX-bf16"

echo "==> vision builds (mlx-vlm)"
for bits in 4 6 8; do
  echo "--- VL ${bits}-bit"
  "$VLM/bin/mlx_vlm.convert" --hf-path "$SRC" -q --q-bits "$bits" \
    --mlx-path "$OUT/Qwen3.8-27B-heretic-VL-MLX-${bits}bit"
done
echo "--- VL bf16"
"$VLM/bin/mlx_vlm.convert" --hf-path "$SRC" \
  --mlx-path "$OUT/Qwen3.8-27B-heretic-VL-MLX-bf16"

echo
echo "==> done. verify with:"
echo "    python scripts/verify_model.py $OUT/Qwen3.8-27B-heretic-VL-MLX-4bit"
