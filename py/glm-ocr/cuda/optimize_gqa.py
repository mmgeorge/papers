"""Manual ONNX graph surgery: replace raw attention ops with GroupQueryAttention.

ORT's automatic `optimize_model(model_type="gpt2")` can't fuse our traced
attention pattern because it's too different from what FusionRotaryAttention
expects (3 separate Q/K/V projections, inline M-RoPE, GQA repeat_kv,
WhereStaticCache). This script does the surgery manually.

## What it does

For each of 16 decoder layers, replaces ~23 nodes (the attention computation
between RoPE output and the output projection MatMul) with a single
`com.microsoft.GroupQueryAttention` (GQA) node.

### Per-layer node replacement

**Removed** (~23 nodes/layer):
- WhereStaticCache KV update (Squeeze, Equal, Reshape, Expand, Where × 2)
- repeat_kv expansion (Unsqueeze, Expand, Reshape × 4 for K and V)
- Scaled dot product attention (Mul × 2, MatMul QK^T, Add mask, Softmax, MatMul attn*V)
- Output reshape (Transpose, Reshape)

**Added** (7 nodes/layer):
- Transpose Q: BNSH [1,16,1,128] → BSNH [1,1,16,128]
- Reshape Q: BSNH → flat [1,1,2048]
- Transpose K: BNSH [1,8,1,128] → BSNH [1,1,8,128]
- Reshape K: BSNH → flat [1,1,1024]
- Transpose V: BNSH [1,8,1,128] → BSNH [1,1,8,128]
- Reshape V: BSNH → flat [1,1,1024]
- GroupQueryAttention node

### GQA configuration

- `do_rotary=0`: RoPE is applied externally (M-RoPE ops remain in the graph)
- `num_heads=16`: query attention heads
- `kv_num_heads=8`: key/value heads (GQA)
- `scale=0.0`: default (1/sqrt(head_size))

### New model inputs

- `seqlens_k`: [1] int32 — total attended length - 1 (= prefill_len + step)
- `total_sequence_length`: [1] int32 — total attended length (= prefill_len + step + 1)

### KV cache with CUDA graphs

GQA handles KV cache internally. With past_present_share_buffer (detected at
runtime when past_key and present_key IoBindings point to same GPU buffer),
GQA does in-place KV updates — compatible with CUDA graph capture/replay.

## Result

Reduces total node count from ~1551 to ~1336 and enables FlashAttention V2
on CUDA.

Usage:
  python optimize_gqa.py --model-dir ../model
  python optimize_gqa.py --input ../model/llm_decoder.onnx --output ../model/llm_decoder_gqa.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Architecture constants (must match export.py)
NUM_LAYERS = 16
NUM_HEADS = 16         # query attention heads
NUM_KV_HEADS = 8       # key/value heads
HEAD_DIM = 128
Q_HIDDEN = 2048        # NUM_HEADS * HEAD_DIM (query projection output)
KV_HIDDEN = 1024       # NUM_KV_HEADS * HEAD_DIM


def apply_gqa_surgery(model_path: Path, output_path: Path, max_seq: int):
    """Replace raw attention ops with GQA nodes in the decoder ONNX model."""
    print(f"Loading {model_path.name} ...")
    model = onnx.load(str(model_path))
    nodes = list(model.graph.node)
    print(f"  Original: {len(nodes)} nodes")

    # Build producer map: tensor_name → (node_index, node)
    producer: dict[str, tuple[int, onnx.NodeProto]] = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            producer[o] = (i, n)

    # ── Identify attention subgraphs ──────────────────────────────────

    layers_info = []
    for layer in range(NUM_LAYERS):
        info = _identify_layer(nodes, producer, layer)
        layers_info.append(info)

    # ── Collect nodes to remove ───────────────────────────────────────

    remove_indices = set()
    for info in layers_info:
        for idx in range(info["remove_start"], info["remove_end"] + 1):
            remove_indices.add(idx)

    print(f"  Removing {len(remove_indices)} nodes ({len(remove_indices) // NUM_LAYERS}/layer)")

    # ── Build new node list ───────────────────────────────────────────

    new_nodes = []
    # Track which layer's GQA nodes to insert after each removal block
    layer_insert_points = {}
    for layer, info in enumerate(layers_info):
        layer_insert_points[info["remove_start"]] = layer

    inserted_layers = set()
    for i, node in enumerate(nodes):
        if i in remove_indices:
            # Check if this is the start of a removal block → insert GQA nodes
            if i in layer_insert_points:
                layer = layer_insert_points[i]
                info = layers_info[layer]
                gqa_nodes = _make_gqa_nodes(layer, info, max_seq)
                new_nodes.extend(gqa_nodes)
                inserted_layers.add(layer)
            continue
        new_nodes.append(node)

    assert len(inserted_layers) == NUM_LAYERS, f"Expected {NUM_LAYERS} layers, inserted {len(inserted_layers)}"

    # ── Add new model inputs ──────────────────────────────────────────

    # seqlens_k: [batch_size] int32 — total_seq_len - 1
    seqlens_k_input = helper.make_tensor_value_info(
        "seqlens_k", TensorProto.INT32, [1]
    )
    model.graph.input.append(seqlens_k_input)

    # total_sequence_length: [1] int32 — total attended length
    total_seq_input = helper.make_tensor_value_info(
        "total_sequence_length", TensorProto.INT32, [1]
    )
    model.graph.input.append(total_seq_input)

    # ── Add shape constants as initializers ───────────────────────────

    # Shape tensors for Reshape ops
    _add_initializer(model, "gqa_q_shape", np.array([1, 1, Q_HIDDEN], dtype=np.int64))
    _add_initializer(model, "gqa_kv_shape", np.array([1, 1, KV_HIDDEN], dtype=np.int64))

    # ── Replace node list ─────────────────────────────────────────────

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # ── Add GQA to opset imports ──────────────────────────────────────

    has_ms_domain = any(
        opset.domain == "com.microsoft" for opset in model.opset_import
    )
    if not has_ms_domain:
        ms_opset = model.opset_import.add()
        ms_opset.domain = "com.microsoft"
        ms_opset.version = 1

    # ── Prune dead nodes ──────────────────────────────────────────────
    # The 4D attention mask computation and WhereStaticCache mask
    # computation are now unused. ORT will prune them automatically
    # during session creation, but we can also remove them here.
    _prune_dead_nodes(model)

    # ── Save ──────────────────────────────────────────────────────────

    print(f"  New: {len(model.graph.node)} nodes")
    print(f"  GQA nodes: {sum(1 for n in model.graph.node if n.op_type == 'GroupQueryAttention')}")

    onnx.save(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
    )

    # Print size
    onnx_size = output_path.stat().st_size / 1024 / 1024
    data_path = Path(str(output_path) + ".data")
    data_size = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
    print(f"  Saved: {onnx_size:.1f} MB graph + {data_size:.1f} MB data")


def _identify_layer(
    nodes: list, producer: dict, layer: int
) -> dict:
    """Identify the attention subgraph boundaries for a given layer.

    Returns dict with:
      q_rotated: tensor name of rotated Q (BNSH format)
      k_rotated: tensor name of rotated K (BNSH format)
      v_tensor: tensor name of V (BNSH format)
      output_proj_input: tensor name consumed by output projection MatMul
      remove_start: first node index to remove
      remove_end: last node index to remove (inclusive)
    """
    pk_name = f"present_key_{layer}"
    pv_name = f"present_value_{layer}"

    pk_idx, pk_node = producer[pk_name]
    pv_idx, pv_node = producer[pv_name]

    # K_rotated: Where.input[1] → Expand.input[0]
    _, k_expand = producer[pk_node.input[1]]
    k_rotated = k_expand.input[0]

    # V: Where.input[1] → Expand.input[0]
    _, v_expand = producer[pv_node.input[1]]
    v_tensor = v_expand.input[0]

    # Find Softmax for this layer (within 30 nodes after Where)
    softmax_idx = None
    for j in range(pk_idx, pk_idx + 30):
        if j < len(nodes) and nodes[j].op_type == "Softmax":
            softmax_idx = j
            break
    assert softmax_idx is not None, f"Softmax not found for layer {layer}"

    # Trace back: Softmax ← Add ← MatMul(QK^T) ← Mul(Q, scale) → Q_rotated
    add_out = nodes[softmax_idx].input[0]
    _, add_node = producer[add_out]
    _, matmul_qkt = producer[add_node.input[0]]
    _, scale_q_mul = producer[matmul_qkt.input[0]]
    q_rotated = scale_q_mul.input[0]

    # Output: Softmax → MatMul(attn*V) → Transpose → Reshape → output_proj_MatMul
    # Reshape is at softmax_idx + 3
    reshape_idx = softmax_idx + 3
    assert nodes[reshape_idx].op_type == "Reshape", \
        f"Expected Reshape at {reshape_idx}, got {nodes[reshape_idx].op_type}"
    output_proj_input = nodes[reshape_idx].output[0]

    # Remove range: from Expand before Where_key through the output Reshape.
    # The Expand feeds into Where for KV cache update — both are replaced by GQA.
    # For layer 0, the mask nodes (Squeeze, Equal, Reshape) that precede Expand
    # will become dead and get cleaned up by _prune_dead_nodes().
    remove_start = pk_idx - 1  # Expand node before Where_key
    remove_end = reshape_idx

    return {
        "q_rotated": q_rotated,
        "k_rotated": k_rotated,
        "v_tensor": v_tensor,
        "output_proj_input": output_proj_input,
        "remove_start": remove_start,
        "remove_end": remove_end,
    }


def _make_gqa_nodes(
    layer: int, info: dict, max_seq: int
) -> list[onnx.NodeProto]:
    """Create the replacement nodes for one attention layer.

    Returns a list of ONNX nodes:
    1. Transpose Q (BNSH → BSNH)
    2. Reshape Q (BSNH → BSH flat)
    3. Transpose K (BNSH → BSNH)
    4. Reshape K (BSNH → BSH flat)
    5. Transpose V (BNSH → BSNH)
    6. Reshape V (BSNH → BSH flat)
    7. GroupQueryAttention node
    """
    L = layer
    q_rotated = info["q_rotated"]
    k_rotated = info["k_rotated"]
    v_tensor = info["v_tensor"]
    output_name = info["output_proj_input"]

    new_nodes = []

    # Q: BNSH [1,16,1,128] → BSNH [1,1,16,128] → flat [1,1,2048]
    q_bsnh = f"gqa_q_bsnh_{L}"
    q_flat = f"gqa_q_flat_{L}"
    new_nodes.append(helper.make_node(
        "Transpose", [q_rotated], [q_bsnh],
        name=f"GQA_TransposeQ_{L}", perm=[0, 2, 1, 3],
    ))
    new_nodes.append(helper.make_node(
        "Reshape", [q_bsnh, "gqa_q_shape"], [q_flat],
        name=f"GQA_ReshapeQ_{L}",
    ))

    # K: BNSH [1,8,1,128] → BSNH [1,1,8,128] → flat [1,1,1024]
    k_bsnh = f"gqa_k_bsnh_{L}"
    k_flat = f"gqa_k_flat_{L}"
    new_nodes.append(helper.make_node(
        "Transpose", [k_rotated], [k_bsnh],
        name=f"GQA_TransposeK_{L}", perm=[0, 2, 1, 3],
    ))
    new_nodes.append(helper.make_node(
        "Reshape", [k_bsnh, "gqa_kv_shape"], [k_flat],
        name=f"GQA_ReshapeK_{L}",
    ))

    # V: BNSH [1,8,1,128] → BSNH [1,1,8,128] → flat [1,1,1024]
    v_bsnh = f"gqa_v_bsnh_{L}"
    v_flat = f"gqa_v_flat_{L}"
    new_nodes.append(helper.make_node(
        "Transpose", [v_tensor], [v_bsnh],
        name=f"GQA_TransposeV_{L}", perm=[0, 2, 1, 3],
    ))
    new_nodes.append(helper.make_node(
        "Reshape", [v_bsnh, "gqa_kv_shape"], [v_flat],
        name=f"GQA_ReshapeV_{L}",
    ))

    # GroupQueryAttention node
    # Inputs (14 total, some empty):
    #  0: query [B,S,D_q]
    #  1: key [B,S,D_kv]
    #  2: value [B,S,D_kv]
    #  3: past_key [B,kv_heads,past_len,head_dim]
    #  4: past_value [B,kv_heads,past_len,head_dim]
    #  5: seqlens_k [B] int32
    #  6: total_sequence_length [1] int32
    #  7-8: cos_cache, sin_cache (empty, do_rotary=0)
    #  9-13: optional (empty)
    gqa_output = f"gqa_output_{L}"
    gqa_node = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            q_flat,                    # 0: query
            k_flat,                    # 1: key
            v_flat,                    # 2: value
            f"past_key_{L}",           # 3: past_key
            f"past_value_{L}",         # 4: past_value
            "seqlens_k",               # 5: seqlens_k
            "total_sequence_length",   # 6: total_sequence_length
            "",                        # 7: cos_cache (empty)
            "",                        # 8: sin_cache (empty)
        ],
        outputs=[
            output_name,               # 0: output [B,S,D_q] → feeds output proj
            f"present_key_{L}",        # 1: present_key
            f"present_value_{L}",      # 2: present_value
        ],
        name=f"GQA_Attention_{L}",
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        kv_num_heads=NUM_KV_HEADS,
        do_rotary=0,
        scale=0.0,  # default: 1/sqrt(head_size)
    )
    new_nodes.append(gqa_node)

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

    # Build producer map: tensor_name → node
    producer_map: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for o in node.output:
            producer_map[o] = node

    # Backward reachability from model outputs
    live_nodes = set()  # set of id(node)
    worklist = []

    # Seed with nodes that produce model outputs
    for name in output_names:
        if name in producer_map:
            node = producer_map[name]
            if id(node) not in live_nodes:
                live_nodes.add(id(node))
                worklist.append(node)

    # BFS backward through inputs
    while worklist:
        node = worklist.pop()
        for inp in node.input:
            if inp and inp in producer_map:
                pred = producer_map[inp]
                if id(pred) not in live_nodes:
                    live_nodes.add(id(pred))
                    worklist.append(pred)

    # Keep only live nodes, preserving original order
    before = len(model.graph.node)
    kept = [n for n in model.graph.node if id(n) in live_nodes]
    pruned = before - len(kept)

    if pruned > 0:
        del model.graph.node[:]
        model.graph.node.extend(kept)
        print(f"  Pruned {pruned} dead nodes")


def main():
    parser = argparse.ArgumentParser(
        description="Replace raw attention ops with GQA nodes in GLM-OCR decoder"
    )
    parser.add_argument("--model-dir", type=str, default="../model",
                        help="Directory containing llm_decoder.onnx")
    parser.add_argument("--input", type=str, default=None,
                        help="Input ONNX model path (default: model-dir/llm_decoder.onnx)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output ONNX model path (default: model-dir/llm_decoder_gqa.onnx)")
    parser.add_argument("--max-seq", type=int, default=512,
                        help="Max sequence length (must match export)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    input_path = Path(args.input) if args.input else model_dir / "llm_decoder.onnx"
    output_path = Path(args.output) if args.output else model_dir / "llm_decoder_gqa.onnx"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    apply_gqa_surgery(input_path, output_path, args.max_seq)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
