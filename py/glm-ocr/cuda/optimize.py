"""Post-export ONNX optimization for GLM-OCR LLM models.

Applies ORT's attention fusion to the raw traced ONNX graphs produced by
export.py. The raw graphs have ~1551 ops per decoder with unfused attention
(15+ ops per layer: MatMul, Reshape, Transpose, Softmax, etc.).

## Optimization levels

1. **MHA (MultiHeadAttention)** — Fuses raw attention ops into the
   `com.microsoft.MultiHeadAttention` contrib op. Works on CUDA, CPU, and
   DirectML. Reduces node count from ~1551 to ~200-300.

2. **GQA (GroupQueryAttention)** — Upgrades MHA nodes to
   `com.microsoft.GroupQueryAttention`, which uses FlashAttention V2 on CUDA.
   O(1) memory scaling for attention — critical for long sequences (full-page
   PDF inference). **CUDA only** — no CPU or DirectML implementation exists.

## M-RoPE handling

GLM-OCR uses 3D M-RoPE (Multi-Resolution Rotary Position Embeddings) from
Qwen2-VL. The rotary embedding computation is already in the traced ONNX graph
as explicit ops. ORT's `FusionRotaryAttention` may or may not absorb these
into the fused MHA/GQA node's built-in `do_rotary` attribute.

If fusion doesn't absorb the RoPE (likely, since M-RoPE is 3D not standard
1D), the external RoPE ops remain in the graph and the fused node gets
`do_rotary=0`. This is the same approach used by onnxruntime-genai's
Qwen2.5-VL builder — confirmed working.

## Usage

  # MHA fusion (all backends):
  python optimize.py --model-dir ../model --target directml

  # GQA fusion (CUDA + FlashAttention):
  python optimize.py --model-dir ../model --target cuda

  # Optimize specific model:
  python optimize.py --model-dir ../model --target cuda --only llm_decoder
"""

import argparse
import sys
from pathlib import Path

import onnx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Architecture constants (must match export.py)
NUM_HEADS = 12        # query attention heads
NUM_KV_HEADS = 8      # key/value heads (GQA)
HIDDEN_SIZE = 1536    # hidden dimension


def count_ops(model_path: Path) -> dict[str, int]:
    """Count op types in an ONNX model (loads graph only, no weights)."""
    model = onnx.load(str(model_path), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def optimize_to_mha(input_path: Path, output_path: Path):
    """Fuse raw attention ops into MultiHeadAttention nodes.

    Uses ORT's optimize_model with model_type="gpt2" which triggers
    FusionRotaryAttention. This pattern-matches the raw attention ops
    (Q/K/V projections, reshape, transpose, softmax, matmul) and replaces
    them with a single com.microsoft.MultiHeadAttention node per layer.

    opt_level=0 means fusion only — no constant folding or other transforms
    that might break the graph structure.
    """
    from onnxruntime.transformers.optimizer import optimize_model

    print(f"  Fusing attention ops → MHA ...")
    model = optimize_model(
        str(input_path),
        model_type="gpt2",
        num_heads=NUM_HEADS,
        hidden_size=HIDDEN_SIZE,
        opt_level=0,
    )

    # Check if MHA nodes were created
    mha_count = sum(1 for n in model.model.graph.node
                    if n.op_type == "MultiHeadAttention")
    total_nodes = len(model.model.graph.node)
    print(f"  Result: {mha_count} MHA nodes, {total_nodes} total nodes")

    if mha_count == 0:
        print("  WARNING: No MHA nodes created. The attention pattern may not")
        print("  match ORT's FusionRotaryAttention expectations. The model will")
        print("  still work but without attention fusion benefits.")

    model.save_model_to_file(str(output_path))
    return mha_count


def upgrade_mha_to_gqa(input_path: Path, output_path: Path):
    """Upgrade MHA nodes to GQA (GroupQueryAttention) for FlashAttention V2.

    Takes an MHA-fused model and replaces MultiHeadAttention nodes with
    GroupQueryAttention nodes. GQA adds:
    - seqlens_k input: cumulative sequence lengths for the KV cache
    - total_sequence_length input: total attended length
    - Flash attention V2 kernel on CUDA

    The attention_mask input is consumed by the GQA conversion to build
    the seqlens_k subgraph automatically.
    """
    from onnxruntime.transformers.convert_generation import replace_mha_with_gqa

    print(f"  Upgrading MHA → GQA (FlashAttention V2) ...")

    model = onnx.load(str(input_path))

    replace_mha_with_gqa(model, "attention_mask", NUM_KV_HEADS)

    # Prune unused nodes after conversion
    from onnxruntime.transformers.onnx_model import OnnxModel
    onnx_model = OnnxModel(model)
    onnx_model.prune_graph()

    gqa_count = sum(1 for n in model.graph.node
                    if n.op_type == "GroupQueryAttention")
    total_nodes = len(model.graph.node)
    print(f"  Result: {gqa_count} GQA nodes, {total_nodes} total nodes")

    onnx.save(
        model, str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
    )
    return gqa_count


def optimize_model_file(input_path: Path, output_dir: Path, target: str):
    """Optimize a single ONNX model file.

    Args:
        input_path: Path to raw exported ONNX model
        output_dir: Directory to write optimized model
        target: "directml" for MHA only, "cuda" for MHA → GQA
    """
    stem = input_path.stem  # e.g., "llm" or "llm_decoder"
    print(f"\nOptimizing {input_path.name} (target={target})")

    # Print before stats
    before = count_ops(input_path)
    print(f"  Before: {sum(before.values())} nodes")

    if target == "directml":
        # MHA fusion only (works on all backends)
        out_path = output_dir / f"{stem}_mha.onnx"
        optimize_to_mha(input_path, out_path)
        _print_size(out_path)

    elif target == "cuda":
        # Step 1: MHA fusion (intermediate)
        mha_path = output_dir / f"{stem}_mha.onnx"
        mha_count = optimize_to_mha(input_path, mha_path)

        if mha_count == 0:
            print("  Skipping GQA upgrade (no MHA nodes to convert)")
            _print_size(mha_path)
            return

        # Step 2: GQA upgrade
        gqa_path = output_dir / f"{stem}_gqa.onnx"
        upgrade_mha_to_gqa(mha_path, gqa_path)
        _print_size(gqa_path)

        # Keep MHA version too (useful for debugging / non-CUDA fallback)
        print(f"  Also kept: {mha_path.name}")

    else:
        raise ValueError(f"Unknown target: {target!r}")


def _print_size(path: Path):
    """Print model file size including external data."""
    onnx_size = path.stat().st_size / 1024 / 1024
    data_path = Path(str(path) + ".data")
    data_size = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
    total = onnx_size + data_size
    if data_size > 0:
        print(f"  Size: {onnx_size:.1f} MB graph + {data_size:.1f} MB data = {total:.1f} MB")
    else:
        print(f"  Size: {total:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize GLM-OCR ONNX models with MHA/GQA fusion"
    )
    parser.add_argument("--model-dir", type=str, default="../model",
                        help="Directory containing exported ONNX models")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as model-dir)")
    parser.add_argument("--target", type=str, required=True,
                        choices=["cuda", "directml"],
                        help="Target backend: 'cuda' for GQA, 'directml' for MHA")
    parser.add_argument("--only", type=str, default=None,
                        choices=["llm", "llm_decoder"],
                        help="Optimize only this model (default: both)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to optimize
    models = []
    if args.only:
        models.append(args.only)
    else:
        models = ["llm", "llm_decoder"]

    for model_name in models:
        input_path = model_dir / f"{model_name}.onnx"
        if not input_path.exists():
            print(f"\nSkipping {model_name}.onnx (not found in {model_dir})")
            continue
        optimize_model_file(input_path, output_dir, args.target)

    print(f"\nDone. Optimized models in: {output_dir}")


if __name__ == "__main__":
    main()
