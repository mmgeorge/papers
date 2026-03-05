"""Export PP-FormulaNet Plus-L to FP32 ONNX for CPU/CoreML inference.

Pipeline:
1. Download PaddlePaddle checkpoint and extract .npz weights
2. Export encoder and decoder as FP32 ONNX

Output: output/encoder.onnx, output/decoder.onnx
"""

import argparse
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from common.weights import convert_and_save
from common.export_common import export_encoder_fp32, export_decoder_fp32


def main():
    parser = argparse.ArgumentParser(description="Export PP-FormulaNet for CPU/CoreML")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for output ONNX models (default: output)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download and extract weights
    print("=" * 60)
    print("Step 1: Extract weights from PaddlePaddle checkpoint")
    print("=" * 60)
    convert_and_save(output_dir)

    # Step 2: Export FP32 ONNX
    print("\n" + "=" * 60)
    print("Step 2: Export FP32 ONNX models")
    print("=" * 60)
    encoder_path = export_encoder_fp32(output_dir, output_dir)
    decoder_path = export_decoder_fp32(output_dir, output_dir)

    print("\n" + "=" * 60)
    print("Done! Final models:")
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
