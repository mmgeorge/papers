"""Run TableFormer ONNX inference and draw cell bboxes on the crop image."""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from common.preprocess import preprocess
from run_onnx import decode, predict_bboxes, cxcywh_to_xyxy, otsl_to_html, CELL_TOKENS, TAG_NAMES


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_bboxes.py <image_path> [model_dir] [output_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data")
    output_path = sys.argv[3] if len(sys.argv) > 3 else "bboxes_python.png"

    # Load ONNX sessions
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    enc_sess = ort.InferenceSession(
        str(model_dir / "encoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    dec_sess = ort.InferenceSession(
        str(model_dir / "decoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    bbox_sess = ort.InferenceSession(
        str(model_dir / "bbox_decoder.onnx"), opts, providers=["CPUExecutionProvider"]
    )

    # Preprocess and decode
    pixel_values = preprocess(image_path)
    tokens, cell_hidden_states, bboxes_to_merge, enc_raw = decode(
        enc_sess, dec_sess, pixel_values
    )

    print(f"Tokens: {len(tokens)}, Hidden states: {len(cell_hidden_states)}")
    print(f"Spans to merge: {bboxes_to_merge}")

    # BBox prediction
    bboxes_cxcywh, classes = predict_bboxes(bbox_sess, enc_raw, cell_hidden_states, bboxes_to_merge)
    bboxes_xyxy = cxcywh_to_xyxy(bboxes_cxcywh)
    print(f"Merged bboxes: {len(bboxes_xyxy)}")

    # Print all bboxes
    for i, bbox in enumerate(bboxes_xyxy):
        print(f"  [{i:3d}] [{bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f}]")

    # Load original image and draw bboxes
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Color cycle
    colors = [
        (0, 200, 0, 80),    # green
        (0, 0, 200, 80),    # blue
        (200, 0, 0, 80),    # red
        (200, 200, 0, 80),  # yellow
        (200, 0, 200, 80),  # magenta
        (0, 200, 200, 80),  # cyan
        (255, 128, 0, 80),  # orange
    ]

    for i, bbox in enumerate(bboxes_xyxy):
        x1 = max(0, bbox[0]) * w
        y1 = max(0, bbox[1]) * h
        x2 = min(1, bbox[2]) * w
        y2 = min(1, bbox[3]) * h
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (200,), width=2)

    result = Image.alpha_composite(img, overlay)
    result.save(output_path)
    print(f"\nSaved bbox overlay to: {output_path}")

    # Also print HTML
    html = otsl_to_html(tokens)
    print(f"\nHTML: {html[:200]}...")


if __name__ == "__main__":
    main()
