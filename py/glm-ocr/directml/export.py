"""Export GLM-OCR for DirectML inference (FP32).

Exports all models in FP32 for DirectML (non-CUDA) inference.
No CUDA-graph decoder is exported — DirectML uses the prefill LLM
(llm.onnx) with session.run() for both prefill and decode steps.

The vision encoder is exported without MHA surgery (raw attention ops).
The LLM can optionally be MHA-fused via optimize.py --target directml.

Key differences from CUDA export:
- FP32 dtype — no BF16 conversion, no FP32 fallback Cast nodes
- No decoder — DirectML can't use CUDA graphs
- No MHA surgery on vision encoder — uses raw vision_encoder.onnx
- MHA on LLM — via optimize.py --target directml (optional)

Output:
  model/vision_encoder.onnx  — CogViT (FP32, no MHA fusion)
  model/embedding.onnx       — Token embeddings (FP32)
  model/llm.onnx             — LLM with dynamic KV cache (FP32)
  model/llm_mha.onnx         — MHA-fused LLM (optional, via --optimize)

Usage:
  uv run python export.py
  uv run python export.py --output-dir output
  uv run python export.py --skip-optimize   # raw export only, no MHA fusion
  uv run python export.py --skip-vision     # skip vision encoder export
"""

import argparse
import subprocess
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
    parser = argparse.ArgumentParser(description="Export GLM-OCR for DirectML (FP32)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Model output directory (default: ../model)")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip MHA optimization (raw export only)")
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

    if args.skip_optimize:
        print("\nSkipping MHA optimization (--skip-optimize)")
        print("Use llm.onnx directly for inference.")
    else:
        # Apply MHA fusion to llm.onnx for DirectML
        print("\n" + "=" * 60)
        print("Applying MHA fusion (DirectML target)")
        print("=" * 60)
        cuda_dir = Path(__file__).parent.parent / "cuda"
        optimize_cmd = [
            sys.executable, str(cuda_dir / "optimize.py"),
            "--model-dir", str(output_dir),
            "--target", "directml",
            "--only", "llm",
        ]
        subprocess.run(optimize_cmd, check=True)

    # Summary
    print(f"\n{'='*60}")
    print("Done! DirectML models (FP32):")
    print(f"  Vision encoder: {output_dir / 'vision_encoder.onnx'}")
    print(f"  Embedding:      {output_dir / 'embedding.onnx'}")
    if not args.skip_optimize:
        print(f"  LLM (MHA):      {output_dir / 'llm_mha.onnx'}")
    else:
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
