"""Export GLM-OCR for DirectML inference.

Pipeline:
1. Run the main export (cuda/export.py) to produce raw ONNX models
2. Apply MHA fusion (cuda/optimize.py --target directml) for fused attention

DirectML uses MultiHeadAttention (MHA) fusion — not GQA, which is CUDA-only.
The CUDA-graph decoder (llm_decoder.onnx) is not exported since DirectML
cannot use CUDA graphs. Inference uses the prefill LLM (llm.onnx) with
session.run() for both prefill and decode steps.

Output:
  model/vision_encoder.onnx  — CogViT (downloaded + FP16 converted)
  model/embedding.onnx       — Token embeddings
  model/llm_mha.onnx         — MHA-fused LLM (used for both prefill and decode)

Usage:
  uv run python export.py
  uv run python export.py --output-dir output
  uv run python export.py --skip-optimize   # raw export only, no MHA fusion
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CUDA_DIR = Path(__file__).parent.parent / "cuda"
DEFAULT_OUTPUT_DIR = "../model"


def main():
    parser = argparse.ArgumentParser(description="Export GLM-OCR for DirectML")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Model output directory (default: ../model)")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip MHA optimization (raw export only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Run the main export (produces raw ONNX models)
    # Skip the CUDA-graph decoder since DirectML can't use it
    print("=" * 60)
    print("Step 1: Export raw ONNX models")
    print("=" * 60)
    export_cmd = [
        sys.executable, str(CUDA_DIR / "export.py"),
        "--output-dir", str(output_dir),
        "--skip-decoder",  # No CUDA-graph decoder for DirectML
    ]
    result = subprocess.run(export_cmd, check=True)

    if args.skip_optimize:
        print("\nSkipping MHA optimization (--skip-optimize)")
        print("Use llm.onnx directly for inference.")
        return

    # Step 2: Apply MHA fusion to llm.onnx
    print("\n" + "=" * 60)
    print("Step 2: Apply MHA fusion (DirectML target)")
    print("=" * 60)
    optimize_cmd = [
        sys.executable, str(CUDA_DIR / "optimize.py"),
        "--model-dir", str(output_dir),
        "--target", "directml",
        "--only", "llm",
    ]
    subprocess.run(optimize_cmd, check=True)

    print("\n" + "=" * 60)
    print("Done! DirectML models:")
    print(f"  Vision encoder: {output_dir / 'vision_encoder.onnx'}")
    print(f"  Embedding:      {output_dir / 'embedding.onnx'}")
    print(f"  LLM (MHA):      {output_dir / 'llm_mha.onnx'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
