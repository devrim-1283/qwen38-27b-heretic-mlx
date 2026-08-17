#!/usr/bin/env python3
"""Generate a test image with known content, so a vision claim can be checked.

The image holds three shapes in fixed positions and two strings. A build that
really sees images names the shapes with the right colours and positions and
reads both strings back exactly. A text-only build converted by mistake will
answer confidently about an image it never received.

    python vision_test.py                          # writes vision-test.png
    mlx_vlm.generate --model <path> --image vision-test.png \
      --prompt "Name the shapes and their colours, then read the text and the code."

Expected: red square (left), blue circle (centre), green triangle (right),
"MLX VISION TEST", and the code "7391-ZQ".
"""
import sys

from PIL import Image, ImageDraw

OUT = sys.argv[1] if len(sys.argv) > 1 else "vision-test.png"

img = Image.new("RGB", (900, 500), "white")
d = ImageDraw.Draw(img)
d.rectangle([60, 60, 300, 300], fill="red")
d.ellipse([380, 60, 620, 300], fill="blue")
d.polygon([(700, 300), (820, 60), (940, 300)], fill="green")
d.text((60, 380), "MLX VISION TEST", fill="black")
d.text((60, 420), "code: 7391-ZQ", fill="black")
img.save(OUT)

print(f"wrote {OUT}")
print("expected: red square left, blue circle centre, green triangle right,")
print("          text 'MLX VISION TEST', code '7391-ZQ'")
