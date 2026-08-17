#!/usr/bin/env python3
"""Check an MLX model directory before trusting or publishing it.

Reports whether every shard the index references exists, whether the declared
size matches what is on disk, whether each safetensors header parses, and
whether the vision tower survived conversion.

That last check is the point: mlx-lm silently produces a text-only build from a
multimodal checkpoint. Nothing errors, the model loads and generates fine, and
images just do not work.

    python verify_model.py ~/models/Qwen3.8-27B-heretic-VL-MLX-4bit
"""
import json
import os
import struct
import sys


def verify(path):
    idx_path = os.path.join(path, "model.safetensors.index.json")
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(idx_path):
        return fail(f"no model.safetensors.index.json in {path}")

    idx = json.load(open(idx_path))
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    weight_map = idx["weight_map"]
    shards = sorted(set(weight_map.values()))

    print(f"path    : {path}")
    print(f"shards  : {len(shards)}")
    print(f"tensors : {len(weight_map)}")

    missing = [s for s in shards if not os.path.exists(os.path.join(path, s))]
    if missing:
        return fail(f"missing shards: {missing}")
    print("missing : none")

    declared = idx.get("metadata", {}).get("total_size", 0)
    on_disk = sum(os.path.getsize(os.path.join(path, s)) for s in shards)
    print(f"size    : {on_disk / 1e9:.1f} GB on disk, {declared / 1e9:.1f} GB declared")

    bad = []
    for s in shards:
        p = os.path.join(path, s)
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            if n <= 0 or 8 + n > os.path.getsize(p):
                bad.append(s)
                continue
            try:
                json.loads(fh.read(n))
            except Exception:
                bad.append(s)
    if bad:
        return fail(f"unparseable safetensors headers: {bad}")
    print("headers : all parse")

    quant = cfg.get("quantization")
    print(f"quant   : {quant if quant else 'none (bf16)'}")

    vision_tensors = sum(1 for k in weight_map if "vis" in k.lower())
    has_vision_cfg = "vision_config" in cfg
    prefixes = sorted({k.split(".")[0] for k in weight_map})
    print(f"prefixes: {prefixes}")
    if has_vision_cfg and vision_tensors:
        print(f"vision  : YES — {vision_tensors} tensors, vision_config present")
    else:
        print(f"vision  : NO — text-only build (vision_config={has_vision_cfg}, "
              f"vision tensors={vision_tensors})")

    print("\nOK")
    return 0


def fail(msg):
    print(f"\nFAIL: {msg}")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(verify(sys.argv[1]))
