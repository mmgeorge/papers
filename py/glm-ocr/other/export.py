"""Export GLM-OCR for CPU/CoreML inference (FP32).

Exports all models in FP32 for non-CUDA inference (CPU, CoreML).
No CUDA-graph decoder is exported — the prefill LLM (llm.onnx) is reused
with session.run() for both prefill and decode steps.

The vision encoder is exported without MHA surgery (raw attention ops).

Key differences from CUDA export:
- FP32 dtype — no BF16 conversion, no FP32 fallback Cast nodes
- No decoder — no CUDA graphs on CPU/CoreML
- No MHA surgery on vision encoder — uses raw vision_encoder.onnx

Output:
  model/vision_encoder.onnx  — CogViT (FP32, no MHA fusion)
  model/embedding.onnx       — Token embeddings (FP32)
  model/llm.onnx             — LLM with dynamic KV cache (FP32)

Usage:
  uv run python export.py
  uv run python export.py --output-dir output
  uv run python export.py --skip-vision     # skip vision encoder export
"""

import argparse
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from common.export import (
    DEFAULT_OUTPUT_DIR,
    load_model,
    export_vision_encoder,
    export_embedding,
    export_llm,
)


def main():
    parser = argparse.ArgumentParser(description="Export GLM-OCR for CPU/CoreML (FP32)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Model output directory (default: ../model)")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip vision encoder export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model in FP32
    model, processor, config = load_model(torch.float32, output_dir)

    with torch.no_grad():
        # Embedding (FP32)
        export_embedding(model, output_dir)

        # LLM with dynamic KV cache (FP32)
        export_llm(model, output_dir, torch.float32)

        # Vision encoder (FP32, no MHA surgery)
        if args.skip_vision:
            print("Skipping vision encoder (--skip-vision)")
        else:
            export_vision_encoder(model, output_dir, torch.float32, apply_mha=False)

    # Summary
    print(f"\n{'='*60}")
    print("Done! FP32 models (CPU/CoreML):")
    print(f"  Vision encoder: {output_dir / 'vision_encoder.onnx'}")
    print(f"  Embedding:      {output_dir / 'embedding.onnx'}")
    print(f"  LLM:            {output_dir / 'llm.onnx'}")

    total = 0
    for f in sorted(output_dir.glob("*.onnx*")):
        if f.suffix in ('.onnx', '.data'):
            sz = f.stat().st_size / 1024 / 1024
            total += sz
            print(f"  {f.name:40s} {sz:>8.1f} MB")
    print(f"  {'TOTAL':40s} {total:>8.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
