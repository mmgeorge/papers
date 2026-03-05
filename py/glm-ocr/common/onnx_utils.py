"""ONNX graph utilities for GLM-OCR model post-processing.

- FP32 → BF16 conversion with FP32 islands for unsupported ops
- Topological sort to fix node ordering after graph surgery
"""

from pathlib import Path

import numpy as np
import onnx


def convert_fp32_to_bf16(onnx_path: Path):
    """Convert an FP32 ONNX graph to BF16 in-place, with FP32 islands.

    Direct FP32→BF16 conversion (no FP16 intermediate). For ops that lack
    BF16 CUDA kernels (Conv, LayerNorm, Range, etc.), weights stay FP32 and
    Cast nodes wrap their inputs/outputs. FP32 preserves full BF16 precision
    (unlike FP16 which has a narrower exponent range).

    This is lossless for native BF16 weights: the FP32 graph was traced
    from BF16 weights (BF16→FP32 upcast is exact), so FP32→BF16 rounds
    back to the original values.
    """
    from ml_dtypes import bfloat16 as ml_bf16

    FLOAT = onnx.TensorProto.FLOAT        # 1
    BFLOAT16 = onnx.TensorProto.BFLOAT16  # 16

    # Ops without BF16 support in ORT 1.24 CUDA EP (ONNX type constraints reject
    # bfloat16). Fall back to FP32 (not FP16) to preserve native BF16 precision.
    # Tested empirically; see PORTING.md for the full compatibility table.
    FP32_OPS = {
        "Conv", "Einsum", "LayerNormalization", "Range",
        "Sin", "Cos", "Reciprocal", "Squeeze", "ReduceMean",
    }

    model = onnx.load(str(onnx_path))

    # 1. Catalog non-BF16 nodes and their input tensor names
    fp32_input_names = set()   # inputs to FP32 ops (initializers stay FP32)
    non_bf16_count = 0
    for node in model.graph.node:
        if node.op_type in FP32_OPS:
            non_bf16_count += 1
            for name in node.input:
                if name:
                    fp32_input_names.add(name)
    print(f"    {non_bf16_count} non-BF16 node(s)")

    # Build maps
    init_map = {init.name: init for init in model.graph.initializer}
    vi_map = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        vi_map[vi.name] = vi

    # 2. Identify initializers consumed ONLY by FP32 ops (not shared with BF16 ops).
    # These stay FP32; shared initializers become BF16 (Cast nodes handle FP32 ops).
    all_input_names = set()
    for node in model.graph.node:
        for name in node.input:
            if name:
                all_input_names.add(name)

    fp32_only_inits = set()
    for name in fp32_input_names:
        if name in init_map:
            # Check if this init is consumed by ANY non-FP32 op
            shared_with_bf16 = False
            for node in model.graph.node:
                if node.op_type not in FP32_OPS and name in list(node.input):
                    shared_with_bf16 = True
                    break
            if not shared_with_bf16:
                fp32_only_inits.add(name)

    n_bf16 = 0
    n_fp32_kept = 0
    for init in model.graph.initializer:
        if init.data_type != FLOAT:
            continue
        if init.name in fp32_only_inits:
            # Exclusively consumed by FP32 ops → keep FP32
            n_fp32_kept += 1
        else:
            # Everything else (including shared) → BF16
            raw = np.frombuffer(init.raw_data, dtype=np.float32)
            init.raw_data = raw.astype(ml_bf16).tobytes()
            init.data_type = BFLOAT16
            n_bf16 += 1

    # 3. Update all FP32 type annotations to BF16
    # (FP32 op outputs get BF16 too — Cast nodes in step 5 bridge the gap)
    n_vi = 0
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.tensor_type.elem_type == FLOAT:
            vi.type.tensor_type.elem_type = BFLOAT16
            n_vi += 1

    # 4. Update Cast nodes: Cast(to=FLOAT) → Cast(to=BF16)
    n_cast = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == FLOAT:
                    attr.i = BFLOAT16
                    n_cast += 1

    print(f"    {n_bf16} init→BF16, {n_fp32_kept} init kept FP32, {n_vi} annotations, {n_cast} Casts")

    # 5. Insert Cast nodes around non-BF16 ops (BF16↔FP32)
    original_nodes = list(model.graph.node)
    new_nodes = []
    # Track already-inserted BF16→FP32 casts to avoid duplicates when
    # multiple FP32 ops share the same input tensor.
    bf16_to_fp32_casts = {}  # inp_name → cast_out_name

    for node in original_nodes:
        if node.op_type not in FP32_OPS:
            new_nodes.append(node)
            continue

        # Cast(BF16→FP32) for each BF16 input (including shared BF16 initializers)
        for i, inp_name in enumerate(node.input):
            if not inp_name:
                continue
            # Skip FP32-only initializers (already FP32, no cast needed)
            if inp_name in fp32_only_inits:
                continue
            # Skip non-BF16 initializers
            if inp_name in init_map and init_map[inp_name].data_type != BFLOAT16:
                continue
            # Check if it's a BF16 tensor (value_info or initializer)
            is_bf16 = False
            if inp_name in vi_map and vi_map[inp_name].type.tensor_type.elem_type == BFLOAT16:
                is_bf16 = True
            elif inp_name in init_map and init_map[inp_name].data_type == BFLOAT16:
                is_bf16 = True
            if not is_bf16:
                continue
            if inp_name in bf16_to_fp32_casts:
                # Reuse existing Cast node
                node.input[i] = bf16_to_fp32_casts[inp_name]
            else:
                cast_out = f"{inp_name}__to_fp32"
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[inp_name], outputs=[cast_out],
                    to=FLOAT, name=f"Cast_bf16_to_fp32_{inp_name}",
                )
                new_vi = model.graph.value_info.add()
                new_vi.name = cast_out
                new_vi.type.tensor_type.elem_type = FLOAT
                if inp_name in vi_map and vi_map[inp_name].type.tensor_type.HasField('shape'):
                    new_vi.type.tensor_type.shape.CopyFrom(
                        vi_map[inp_name].type.tensor_type.shape)
                node.input[i] = cast_out
                new_nodes.append(cast_node)
                bf16_to_fp32_casts[inp_name] = cast_out

        new_nodes.append(node)

        # Cast(FP32→BF16) for each output so downstream BF16 nodes get BF16 data
        for i, out_name in enumerate(node.output):
            if not out_name:
                continue
            tmp_name = f"{out_name}__fp32_out"
            # Create FP32 value_info for the op's actual output
            new_vi = model.graph.value_info.add()
            new_vi.name = tmp_name
            new_vi.type.tensor_type.elem_type = FLOAT
            if out_name in vi_map and vi_map[out_name].type.tensor_type.HasField('shape'):
                new_vi.type.tensor_type.shape.CopyFrom(
                    vi_map[out_name].type.tensor_type.shape)
            cast_node = onnx.helper.make_node(
                "Cast", inputs=[tmp_name], outputs=[out_name],
                to=BFLOAT16, name=f"Cast_fp32_to_bf16_{out_name}",
            )
            node.output[i] = tmp_name
            new_nodes.append(cast_node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    # Delete old external data file before saving to avoid stale/appended data.
    data_path = Path(str(onnx_path) + ".data")
    if data_path.exists():
        data_path.unlink()

    # Externalize tensors >= 1 KB to avoid padding overhead on tiny tensors.
    # convert_attribute=True ensures Constant node data is also saved externally.
    onnx.save(
        model, str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_path.name + ".data",
        size_threshold=1024,
        convert_attribute=True,
    )


def topological_sort(graph):
    """Fix ONNX node ordering after inserting Cast nodes.

    When Cast nodes are inserted around FP32-fallback ops, they may break
    topological order. ORT requires nodes to be ordered such that all
    inputs are defined before they're consumed. This function performs
    a DFS-based topological sort to restore a valid ordering.
    """
    producers = {}
    for node in graph.node:
        for out in node.output:
            producers[out] = node

    input_names = set()
    for inp in graph.input:
        input_names.add(inp.name)
    for init in graph.initializer:
        input_names.add(init.name)

    visited = set()
    ordered = []

    def visit(node):
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for inp_name in node.input:
            if inp_name in producers and id(producers[inp_name]) not in visited:
                visit(producers[inp_name])
        ordered.append(node)

    for node in graph.node:
        visit(node)

    del graph.node[:]
    graph.node.extend(ordered)
