"""Manual ONNX graph surgery: replace raw attention ops with MultiHeadAttention.

ORT's automatic `optimize_model(model_type="vit")` can't fuse the vision encoder
attention because Q/K RMSNorm and 2D RoPE between projection and attention break
pattern matching. This script does the surgery manually.

## What it does

For each of 24 vision transformer layers, replaces ~25 nodes (the attention
computation between Q/K RoPE output and the output projection Gemm) with a
single `com.microsoft.MultiHeadAttention` (MHA) node plus reshape ops.

### Per-layer node replacement

**Removed** (~25 nodes/layer):
- Q/K/V transpose + unsqueeze (reshaping to [1, heads, seq, dim])
- K transpose for QK^T
- MatMul(Q@K^T), scale Mul, Softmax, Cast(s), MatMul(attn@V)
- Output squeeze, transpose, reshape (back to [seq, hidden])
- Associated Constant nodes

**Added** (5 nodes/layer):
- Reshape Q: [seq, 16, 64] → [1, seq, 1024]
- Reshape K: [seq, 16, 64] → [1, seq, 1024]
- Reshape V: [seq, 16, 64] → [1, seq, 1024]
- MultiHeadAttention node → [1, seq, 1024]
- Reshape output: [1, seq, 1024] → [seq, 1024] (for proj Gemm)

### MHA configuration

- `num_heads=16`: attention heads (1024 / 64)
- `scale=0.125`: 1/sqrt(64)
- `unidirectional=0`: bidirectional attention (vision encoder, not causal)

## Result

Reduces total node count from ~3663 to ~3183 and enables FlashAttention on CUDA.

Usage:
  python optimize_mha_vision.py --model-dir ../model
  python optimize_mha_vision.py --input ../model/vision_encoder.onnx --output ../model/vision_encoder_mha.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Architecture constants
NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 1024


def apply_mha_surgery(model_path: Path, output_path: Path):
    """Replace raw attention ops with MHA nodes in the vision encoder ONNX model."""
    print(f"Loading {model_path.name} ...")
    model = onnx.load(str(model_path))
    nodes = list(model.graph.node)
    print(f"  Original: {len(nodes)} nodes")

    # Build producer map: tensor_name → node
    producer: dict[str, onnx.NodeProto] = {}
    for n in nodes:
        for o in n.output:
            producer[o] = n

    # Build consumer map: tensor_name → [nodes]
    consumer: dict[str, list[onnx.NodeProto]] = {}
    for n in nodes:
        for inp in n.input:
            if inp:
                consumer.setdefault(inp, []).append(n)

    # ── Find all Softmax nodes (one per layer) ───────────────────────
    softmaxes = [n for n in nodes if n.op_type == "Softmax"]
    assert len(softmaxes) == NUM_LAYERS, \
        f"Expected {NUM_LAYERS} Softmax nodes, found {len(softmaxes)}"

    # ── Identify attention subgraphs ─────────────────────────────────
    layers_info = []
    for layer, sm_node in enumerate(softmaxes):
        info = _identify_layer(producer, consumer, sm_node, layer)
        layers_info.append(info)

    # ── Collect nodes to remove ──────────────────────────────────────
    all_remove = set()
    for info in layers_info:
        all_remove.update(id(n) for n in info["nodes_to_remove"])

    print(f"  Removing {len(all_remove)} nodes ({len(all_remove) // NUM_LAYERS}/layer)")

    # ── Build new node list ──────────────────────────────────────────
    # Insert MHA nodes right where we encounter the first removed node per layer
    layer_first_removed = {}
    for layer, info in enumerate(layers_info):
        for n in nodes:
            if id(n) in all_remove and id(n) in {id(x) for x in info["nodes_to_remove"]}:
                if layer not in layer_first_removed:
                    layer_first_removed[id(n)] = layer
                break

    # Simpler approach: insert MHA nodes at the position of each Softmax
    softmax_ids = {id(sm): layer for layer, sm in enumerate(softmaxes)}

    new_nodes = []
    inserted_layers = set()
    for node in nodes:
        if id(node) in all_remove:
            # At the Softmax position, insert MHA replacement nodes
            if id(node) in softmax_ids:
                layer = softmax_ids[id(node)]
                info = layers_info[layer]
                mha_nodes = _make_mha_nodes(layer, info)
                new_nodes.extend(mha_nodes)
                inserted_layers.add(layer)
            continue
        new_nodes.append(node)

    assert len(inserted_layers) == NUM_LAYERS, \
        f"Expected {NUM_LAYERS} layers, inserted {len(inserted_layers)}"

    # ── Add shape constants as initializers ──────────────────────────
    _add_initializer(model, "mha_qkv_shape_3d", np.array([1, -1, HIDDEN], dtype=np.int64))
    _add_initializer(model, "mha_output_shape_2d", np.array([-1, HIDDEN], dtype=np.int64))

    # ── Replace node list ────────────────────────────────────────────
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # ── Add com.microsoft opset import ───────────────────────────────
    has_ms_domain = any(
        opset.domain == "com.microsoft" for opset in model.opset_import
    )
    if not has_ms_domain:
        ms_opset = model.opset_import.add()
        ms_opset.domain = "com.microsoft"
        ms_opset.version = 1

    # ── Prune dead nodes ─────────────────────────────────────────────
    _prune_dead_nodes(model)

    # ── Save ─────────────────────────────────────────────────────────
    print(f"  New: {len(model.graph.node)} nodes")
    mha_count = sum(1 for n in model.graph.node if n.op_type == "MultiHeadAttention")
    print(f"  MHA nodes: {mha_count}")

    onnx.save(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
    )

    onnx_size = output_path.stat().st_size / 1024 / 1024
    data_path = Path(str(output_path) + ".data")
    data_size = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
    print(f"  Saved: {onnx_size:.1f} MB graph + {data_size:.1f} MB data")


def _identify_layer(
    producer: dict, consumer: dict, sm_node: onnx.NodeProto, layer: int
) -> dict:
    """Identify the attention subgraph for a given layer.

    Traces backward and forward from the Softmax node to find:
    - Q, K, V input tensors (post-RoPE / post-split)
    - Output tensor name (feeding into proj Gemm)
    - All nodes in the attention core to remove
    """
    nodes_to_remove = []

    # ── Backward: Softmax ← Mul(scale) ← MatMul(Q@K^T) ─────────────
    scale_node = producer[sm_node.input[0]]
    qkt_node = producer[scale_node.input[0]]

    # Q path: MatMul.input[0] ← Unsqueeze ← Transpose ← Q_rope
    q_unsqueeze = producer[qkt_node.input[0]]
    q_transpose = producer[q_unsqueeze.input[0]]
    q_rope = q_transpose.input[0]  # Keep: Q after RoPE [seq, 16, 64]

    # K path: MatMul.input[1] ← Transpose(QK) ← Unsqueeze ← Transpose ← K_rope
    k_transpose_qk = producer[qkt_node.input[1]]
    k_unsqueeze = producer[k_transpose_qk.input[0]]
    k_transpose = producer[k_unsqueeze.input[0]]
    k_rope = k_transpose.input[0]  # Keep: K after RoPE [seq, 16, 64]

    nodes_to_remove.extend([
        q_transpose, q_unsqueeze,
        k_transpose, k_unsqueeze, k_transpose_qk,
        qkt_node, scale_node, sm_node,
    ])

    # ── Forward: Softmax → Cast → Cast → MatMul(attn@V) ─────────────
    cast1 = _sole_consumer(consumer, sm_node.output[0])
    cast2 = _sole_consumer(consumer, cast1.output[0])
    matmul_v = _sole_consumer(consumer, cast2.output[0])

    # V path: MatMul_1.input[1] ← Unsqueeze ← Transpose ← V_tensor
    v_unsqueeze = producer[matmul_v.input[1]]
    v_transpose = producer[v_unsqueeze.input[0]]
    v_tensor = v_transpose.input[0]  # Keep: V after split+squeeze [seq, 16, 64]

    nodes_to_remove.extend([cast1, cast2, matmul_v, v_transpose, v_unsqueeze])

    # ── Continue forward: MatMul_1 → Squeeze → Transpose → Reshape ──
    squeeze = _sole_consumer(consumer, matmul_v.output[0])
    transpose_out = _sole_consumer(consumer, squeeze.output[0])

    # Find the Reshape (may have shape computation nodes in between)
    reshape = None
    for n in consumer.get(transpose_out.output[0], []):
        if n.op_type == "Reshape":
            reshape = n
            break

    if reshape is None:
        raise RuntimeError(
            f"Layer {layer}: Reshape not found after Transpose {transpose_out.name}"
        )

    output_name = reshape.output[0]  # Feeds into proj/Gemm
    nodes_to_remove.extend([squeeze, transpose_out, reshape])

    print(f"  Layer {layer}: Q={q_rope}, K={k_rope}, V={v_tensor} → {output_name}")

    return {
        "q_rope": q_rope,
        "k_rope": k_rope,
        "v_tensor": v_tensor,
        "output_name": output_name,
        "nodes_to_remove": nodes_to_remove,
    }


def _sole_consumer(consumer: dict, tensor_name: str) -> onnx.NodeProto:
    """Get the single consumer of a tensor, asserting there is exactly one."""
    consumers = consumer.get(tensor_name, [])
    assert len(consumers) == 1, \
        f"Expected 1 consumer of {tensor_name}, found {len(consumers)}"
    return consumers[0]


def _make_mha_nodes(layer: int, info: dict) -> list[onnx.NodeProto]:
    """Create replacement nodes for one attention layer.

    Returns:
    1. Reshape Q: [seq, 16, 64] → [1, seq, 1024]
    2. Reshape K: [seq, 16, 64] → [1, seq, 1024]
    3. Reshape V: [seq, 16, 64] → [1, seq, 1024]
    4. MultiHeadAttention node → [1, seq, 1024]
    5. Reshape output: [1, seq, 1024] → [seq, 1024]
    """
    L = layer
    q_rope = info["q_rope"]
    k_rope = info["k_rope"]
    v_tensor = info["v_tensor"]
    output_name = info["output_name"]

    new_nodes = []

    # Reshape Q: [seq, 16, 64] → [1, seq, 1024]
    q_3d = f"mha_q_3d_{L}"
    new_nodes.append(helper.make_node(
        "Reshape", [q_rope, "mha_qkv_shape_3d"], [q_3d],
        name=f"MHA_ReshapeQ_{L}",
    ))

    # Reshape K: [seq, 16, 64] → [1, seq, 1024]
    k_3d = f"mha_k_3d_{L}"
    new_nodes.append(helper.make_node(
        "Reshape", [k_rope, "mha_qkv_shape_3d"], [k_3d],
        name=f"MHA_ReshapeK_{L}",
    ))

    # Reshape V: [seq, 16, 64] → [1, seq, 1024]
    v_3d = f"mha_v_3d_{L}"
    new_nodes.append(helper.make_node(
        "Reshape", [v_tensor, "mha_qkv_shape_3d"], [v_3d],
        name=f"MHA_ReshapeV_{L}",
    ))

    # MultiHeadAttention node
    # Inputs: query, key, value, bias(empty), key_padding_mask(empty),
    #         attention_bias(empty), past_key(empty), past_value(empty)
    mha_output_3d = f"mha_output_3d_{L}"
    mha_node = helper.make_node(
        "MultiHeadAttention",
        inputs=[
            q_3d,   # 0: query [1, seq, 1024]
            k_3d,   # 1: key [1, seq, 1024]
            v_3d,   # 2: value [1, seq, 1024]
            "",     # 3: bias (empty)
            "",     # 4: key_padding_mask (empty)
            "",     # 5: attention_bias (empty)
        ],
        outputs=[
            mha_output_3d,  # 0: output [1, seq, 1024]
        ],
        name=f"MHA_Attention_{L}",
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        scale=1.0 / (HEAD_DIM ** 0.5),  # 0.125
    )
    new_nodes.append(mha_node)

    # Reshape output: [1, seq, 1024] → [seq, 1024] for proj Gemm
    new_nodes.append(helper.make_node(
        "Reshape", [mha_output_3d, "mha_output_shape_2d"], [output_name],
        name=f"MHA_ReshapeOut_{L}",
    ))

    return new_nodes


def _add_initializer(model, name: str, value: np.ndarray):
    """Add a constant tensor as an initializer to the model."""
    tensor = helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=value.shape,
        vals=value.flatten().tolist(),
    )
    model.graph.initializer.append(tensor)


def _prune_dead_nodes(model):
    """Remove nodes whose outputs are not consumed by any other node or model output.

    Uses backward reachability: starts from model outputs and marks all
    nodes that contribute to them as live. Everything else is dead.
    """
    output_names = {o.name for o in model.graph.output}

    producer_map: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for o in node.output:
            producer_map[o] = node

    live_nodes = set()
    worklist = []

    for name in output_names:
        if name in producer_map:
            node = producer_map[name]
            if id(node) not in live_nodes:
                live_nodes.add(id(node))
                worklist.append(node)

    while worklist:
        node = worklist.pop()
        for inp in node.input:
            if inp and inp in producer_map:
                pred = producer_map[inp]
                if id(pred) not in live_nodes:
                    live_nodes.add(id(pred))
                    worklist.append(pred)

    before = len(model.graph.node)
    kept = [n for n in model.graph.node if id(n) in live_nodes]
    pruned = before - len(kept)

    if pruned > 0:
        del model.graph.node[:]
        model.graph.node.extend(kept)
        print(f"  Pruned {pruned} dead nodes")


def main():
    parser = argparse.ArgumentParser(
        description="Replace raw attention ops with MHA nodes in GLM-OCR vision encoder"
    )
    parser.add_argument("--model-dir", type=str, default="../model",
                        help="Directory containing vision_encoder.onnx")
    parser.add_argument("--input", type=str, default=None,
                        help="Input ONNX model path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX model path")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input) if args.input else model_dir / "vision_encoder.onnx"
    output_path = Path(args.output) if args.output else model_dir / "vision_encoder_mha.onnx"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    apply_mha_surgery(input_path, output_path)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
