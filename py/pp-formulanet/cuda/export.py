"""Export PP-FormulaNet Plus-L to optimized ONNX for CUDA inference.

Pipeline:
1. Download PaddlePaddle checkpoint and extract .npz weights
2. Export encoder and decoder as FP32 ONNX
3. Convert both to FP16 (native I/O, no boundary Cast nodes)
4. Add GPU-side ArgMax to decoder (avoids 100KB D2H logits transfer)
5. Clean up intermediate files

Output: output/encoder_fp16.onnx, output/decoder_fp16_argmax.onnx
"""

import argparse
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import onnx
from onnx import helper, TensorProto
from onnxruntime.transformers import float16

from common.weights import convert_and_save
from common.export_common import export_encoder_fp32, export_decoder_fp32


def topological_sort(graph):
    """Fix node ordering after float16 conversion inserts Cast nodes."""
    known = set()
    for inp in graph.input:
        known.add(inp.name)
    for init in graph.initializer:
        known.add(init.name)

    nodes = list(graph.node)
    sorted_nodes = []
    remaining = list(range(len(nodes)))

    while remaining:
        progress = False
        next_remaining = []
        for idx in remaining:
            node = nodes[idx]
            if all(inp == "" or inp in known for inp in node.input):
                sorted_nodes.append(node)
                for out in node.output:
                    known.add(out)
                progress = True
            else:
                next_remaining.append(idx)
        remaining = next_remaining
        if not progress:
            for idx in remaining:
                sorted_nodes.append(nodes[idx])
            break

    del graph.node[:]
    graph.node.extend(sorted_nodes)


def convert_fp16(input_path, output_path, keep_io_types=False):
    """Convert an ONNX model from FP32 to FP16."""
    print(f"\nConverting {input_path} -> {output_path}")
    model = onnx.load(input_path)

    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
    )

    topological_sort(model_fp16.graph)

    onnx.save(model_fp16, output_path)
    onnx.checker.check_model(onnx.load(output_path))

    orig_mb = os.path.getsize(input_path) / 1e6
    fp16_mb = os.path.getsize(output_path) / 1e6
    print(f"  {orig_mb:.1f} MB -> {fp16_mb:.1f} MB ({fp16_mb/orig_mb:.0%})")


def add_argmax_output(input_path, output_path):
    """Add ArgMax + Reshape to decoder for GPU-side token selection."""
    print(f"\nAdding argmax: {input_path} -> {output_path}")
    model = onnx.load(input_path)
    graph = model.graph

    logits_output = None
    for out in graph.output:
        if out.name == "logits":
            logits_output = out
            break
    if logits_output is None:
        raise ValueError("No 'logits' output found in model")

    # ArgMax(logits, axis=2, keepdims=0) -> [1, 1]
    argmax_node = helper.make_node(
        "ArgMax",
        inputs=["logits"],
        outputs=["next_token_2d"],
        axis=2,
        keepdims=0,
    )

    # Reshape to [1] for easy D2H copy
    shape_const = helper.make_tensor(
        "argmax_shape", TensorProto.INT64, [1], [1]
    )
    graph.initializer.append(shape_const)

    reshape_node = helper.make_node(
        "Reshape",
        inputs=["next_token_2d", "argmax_shape"],
        outputs=["next_token"],
    )

    graph.node.append(argmax_node)
    graph.node.append(reshape_node)

    next_token_output = helper.make_tensor_value_info(
        "next_token", TensorProto.INT64, [1]
    )
    graph.output.append(next_token_output)

    onnx.save(model, output_path)
    onnx.checker.check_model(onnx.load(output_path))

    orig_mb = os.path.getsize(input_path) / 1e6
    new_mb = os.path.getsize(output_path) / 1e6
    print(f"  {orig_mb:.1f} MB -> {new_mb:.1f} MB")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")


def main():
    parser = argparse.ArgumentParser(description="Export PP-FormulaNet for CUDA")
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
    encoder_fp32 = export_encoder_fp32(output_dir, output_dir)
    decoder_fp32 = export_decoder_fp32(output_dir, output_dir)

    # Step 3: Convert to FP16 (native I/O)
    print("\n" + "=" * 60)
    print("Step 3: Convert to FP16")
    print("=" * 60)
    encoder_fp16 = os.path.join(output_dir, "encoder_fp16.onnx")
    decoder_fp16 = os.path.join(output_dir, "decoder_fp16.onnx")
    convert_fp16(encoder_fp32, encoder_fp16, keep_io_types=False)
    convert_fp16(decoder_fp32, decoder_fp16, keep_io_types=False)

    # Step 4: Add argmax to decoder
    print("\n" + "=" * 60)
    print("Step 4: Add GPU-side ArgMax to decoder")
    print("=" * 60)
    decoder_final = os.path.join(output_dir, "decoder_fp16_argmax.onnx")
    add_argmax_output(decoder_fp16, decoder_final)

    # Step 5: Clean up intermediates
    print("\n" + "=" * 60)
    print("Step 5: Clean up intermediate files")
    print("=" * 60)
    for path in [encoder_fp32, decoder_fp32, decoder_fp16]:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed: {path}")

    print("\n" + "=" * 60)
    print("Done! Final models:")
    print(f"  Encoder: {encoder_fp16}")
    print(f"  Decoder: {decoder_final}")
    print("=" * 60)


if __name__ == "__main__":
    main()
