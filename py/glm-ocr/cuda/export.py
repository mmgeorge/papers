"""Export GLM-OCR (zai-org/GLM-OCR) to 3-part ONNX.

The model is natively BF16. Default export is FP16 (best ORT compatibility).
BF16 mode exports everything in BF16 (requires CUDA). The vision encoder's
Conv ops stay in FP16 (ORT has no BF16 Conv kernel) with Cast wrappers.

Vision encoder can't be re-exported from PyTorch (M-RoPE uses data-dependent
shapes). Instead, we download the FP32 one from ningpp/GLM-OCR and convert
it using onnxconverter_common with node_block_list for Cast nodes.

Produces (in ./model/):
  - vision_encoder.onnx   (FP16 or BF16 with FP16 Conv islands)
  - embedding.onnx        (FP16 or BF16)
  - llm.onnx              (FP16 or BF16, with KV cache)

Usage:
  python export.py                       # FP16 export (default)
  python export.py --bf16                # BF16 export (CUDA only)
  python export.py --fp32                # FP32 export (debug)
  python export.py --skip-vision         # skip vision encoder
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Monkey-patch transformers 5.2.0 video_processing_auto bug
import importlib
import transformers.models.auto.video_processing_auto as _vpa

def _patched_video_processor_class_from_name(class_name):
    for module_name, extractors in _vpa.VIDEO_PROCESSOR_MAPPING_NAMES.items():
        if extractors is None:
            continue
        if class_name in extractors:
            mn = _vpa.model_type_to_module_name(module_name)
            module = importlib.import_module(f".{mn}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue
    for extractor in _vpa.VIDEO_PROCESSOR_MAPPING._extra_content.values():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)
    return None

_vpa.video_processor_class_from_name = _patched_video_processor_class_from_name

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxconverter_common import float16
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    DynamicCache,
)

MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_OUTPUT_DIR = "../model"

# From config.json
NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1536
VOCAB_SIZE = 59392


# ── Wrapper modules ──────────────────────────────────────────────────────


class EmbedTokensWrapper(nn.Module):
    """Wraps the text embedding layer."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.language_model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLMWrapper(nn.Module):
    """Wraps the LLM decoder with explicit KV cache inputs/outputs.

    Uses explicit named parameters (not *args) for torch.onnx.export
    compatibility with torch 2.10's dynamic_axes handling.
    """

    def __init__(self, model):
        super().__init__()
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds, attention_mask, position_ids,
                past_key_0, past_value_0, past_key_1, past_value_1,
                past_key_2, past_value_2, past_key_3, past_value_3,
                past_key_4, past_value_4, past_key_5, past_value_5,
                past_key_6, past_value_6, past_key_7, past_value_7,
                past_key_8, past_value_8, past_key_9, past_value_9,
                past_key_10, past_value_10, past_key_11, past_value_11,
                past_key_12, past_value_12, past_key_13, past_value_13,
                past_key_14, past_value_14, past_key_15, past_value_15):
        past_keys = [past_key_0, past_key_1, past_key_2, past_key_3,
                     past_key_4, past_key_5, past_key_6, past_key_7,
                     past_key_8, past_key_9, past_key_10, past_key_11,
                     past_key_12, past_key_13, past_key_14, past_key_15]
        past_values = [past_value_0, past_value_1, past_value_2, past_value_3,
                       past_value_4, past_value_5, past_value_6, past_value_7,
                       past_value_8, past_value_9, past_value_10, past_value_11,
                       past_value_12, past_value_13, past_value_14, past_value_15]

        past_key_values = DynamicCache()
        for layer in range(NUM_LAYERS):
            past_key_values.update(past_keys[layer], past_values[layer], layer)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        logits = self.lm_head(outputs.last_hidden_state)
        new_past = outputs.past_key_values

        result = [logits]
        for layer in range(NUM_LAYERS):
            result.append(new_past.layers[layer].keys)
            result.append(new_past.layers[layer].values)
        return tuple(result)


# ── Export functions ──────────────────────────────────────────────────────


def export_vision_encoder(output_dir: Path, precision: str = "fp16"):
    """Download vision encoder from ningpp/GLM-OCR and convert to target precision.

    The vision encoder can't be re-exported from PyTorch because M-RoPE uses
    data-dependent shapes (torch.split with dynamic lengths) that torch.export
    can't trace. We use the community FP32 export and convert.

    For FP16: blocks all Cast-to-FLOAT nodes to preserve type correctness.
    For BF16: converts FP16 to BF16 with FP16 islands around Conv ops
    (ORT has no BF16 Conv kernel on any provider).

    Note: ORT_ENABLE_ALL optimization level breaks with the mixed-type graph
    from blocking Cast nodes. Use ORT_ENABLE_EXTENDED or lower at runtime.
    """
    fp32_path = output_dir / "vision_encoder_fp32.onnx"
    output_path = output_dir / "vision_encoder.onnx"

    # Download FP32 from ningpp/GLM-OCR
    if not fp32_path.exists() and not output_path.exists():
        print("Downloading vision_encoder.onnx from ningpp/GLM-OCR ...")
        from huggingface_hub import hf_hub_download
        hf_hub_download("ningpp/GLM-OCR", "vision_encoder.onnx",
                        local_dir=str(output_dir))
        # Rename to _fp32 so we can save the converted one as the main file
        (output_dir / "vision_encoder.onnx").rename(fp32_path)
        print(f"  Downloaded: {fp32_path.stat().st_size / 1024 / 1024:.0f} MB")
    elif not fp32_path.exists() and output_path.exists():
        print(f"vision_encoder.onnx already present ({output_path.stat().st_size / 1024 / 1024:.0f} MB)")
        return output_path

    if precision == "fp32":
        fp32_path.rename(output_path)
        _print_size(output_path)
        return output_path

    # Convert FP32 → FP16 (intermediate step for both fp16 and bf16)
    print("Converting vision_encoder to FP16 ...")
    model = onnx.load(str(fp32_path))

    # Block all Cast-to-FLOAT nodes: these upcast for ops (Conv, Range,
    # LayerNorm) that require FP32. Changing their target type breaks
    # downstream type constraints.
    block_list = []
    for n in model.graph.node:
        if n.op_type == "Cast":
            for attr in n.attribute:
                if attr.name == "to" and attr.i == 1:  # TensorProto.FLOAT
                    block_list.append(n.name)
    print(f"  Blocking {len(block_list)} Cast-to-FLOAT nodes")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
            node_block_list=block_list,
        )

    if precision == "bf16":
        print("  Converting FP16 → BF16 (Conv ops stay FP16) ...")
        model_out = _convert_vision_fp16_to_bf16(model_fp16)
    else:
        model_out = model_fp16

    onnx.save(
        model_out, str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="vision_encoder.onnx.data",
    )
    _print_size(output_path)

    # Clean up FP32 file
    fp32_path.unlink()
    return output_path


def _convert_vision_fp16_to_bf16(model):
    """Post-process FP16 vision encoder to BF16, with FP16 islands around Conv.

    Conv has no BF16 kernel in ORT, so Conv nodes stay in FP16 with
    Cast(BF16->FP16) before inputs and Cast(FP16->BF16) after outputs.

    Steps:
    1. Convert all FP16 initializers, type annotations, and Cast nodes to BF16
    2. Revert Conv weight/bias initializers back to FP16
    3. Insert Cast wrappers around Conv nodes
    """
    from ml_dtypes import bfloat16 as ml_bf16

    FLOAT16 = onnx.TensorProto.FLOAT16    # 10
    BFLOAT16 = onnx.TensorProto.BFLOAT16  # 16

    # 1. Catalog Conv nodes and their I/O tensor names
    conv_input_names = set()
    conv_output_names = set()
    for node in model.graph.node:
        if node.op_type == "Conv":
            for name in node.input:
                if name:
                    conv_input_names.add(name)
            for name in node.output:
                conv_output_names.add(name)
    print(f"    Found {len(conv_output_names)} Conv node(s) to wrap")

    # 2. Convert ALL FP16 initializers to BF16
    init_map = {}
    n_init = 0
    for init in model.graph.initializer:
        init_map[init.name] = init
        if init.data_type == FLOAT16:
            raw = np.frombuffer(init.raw_data, dtype=np.float16)
            init.raw_data = raw.astype(np.float32).astype(ml_bf16).tobytes()
            init.data_type = BFLOAT16
            n_init += 1

    # 3. Update ALL FP16 type annotations to BF16
    vi_map = {}
    n_vi = 0
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        vi_map[vi.name] = vi
        if vi.type.tensor_type.elem_type == FLOAT16:
            vi.type.tensor_type.elem_type = BFLOAT16
            n_vi += 1

    # 4. Update ALL Cast(to=FP16) to Cast(to=BF16)
    n_cast = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == FLOAT16:
                    attr.i = BFLOAT16
                    n_cast += 1
    print(f"    Converted {n_init} initializers, {n_vi} type annotations, {n_cast} Cast nodes")

    # 5. Fix Conv nodes: keep them in FP16 with Cast wrappers
    # 5a. Convert Conv weight/bias initializers back to FP16
    for name in conv_input_names:
        if name in init_map and init_map[name].data_type == BFLOAT16:
            init = init_map[name]
            raw = np.frombuffer(init.raw_data, dtype=ml_bf16)
            init.raw_data = raw.astype(np.float32).astype(np.float16).tobytes()
            init.data_type = FLOAT16
            if name in vi_map:
                vi_map[name].type.tensor_type.elem_type = FLOAT16

    # 5b/5c. Insert Cast nodes around Conv ops
    # Rebuild node list to handle insertions cleanly
    original_nodes = list(model.graph.node)
    new_nodes = []

    for node in original_nodes:
        if node.op_type != "Conv":
            new_nodes.append(node)
            continue

        # Insert Cast(BF16->FP16) for each non-initializer BF16 input
        for i, inp_name in enumerate(node.input):
            if not inp_name or inp_name in init_map:
                continue
            if inp_name in vi_map and vi_map[inp_name].type.tensor_type.elem_type == BFLOAT16:
                cast_out = f"{inp_name}__to_fp16"
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[inp_name], outputs=[cast_out],
                    to=FLOAT16, name=f"Cast_bf16_to_fp16_{inp_name}",
                )
                new_vi = model.graph.value_info.add()
                new_vi.name = cast_out
                new_vi.type.tensor_type.elem_type = FLOAT16
                if vi_map[inp_name].type.tensor_type.HasField('shape'):
                    new_vi.type.tensor_type.shape.CopyFrom(
                        vi_map[inp_name].type.tensor_type.shape)
                node.input[i] = cast_out
                new_nodes.append(cast_node)

        # The Conv node itself
        new_nodes.append(node)

        # Insert Cast(FP16->BF16) for each BF16-typed output
        for i, out_name in enumerate(node.output):
            if out_name in vi_map and vi_map[out_name].type.tensor_type.elem_type == BFLOAT16:
                fp16_name = f"{out_name}__conv_fp16"
                new_vi = model.graph.value_info.add()
                new_vi.name = fp16_name
                new_vi.type.tensor_type.elem_type = FLOAT16
                if vi_map[out_name].type.tensor_type.HasField('shape'):
                    new_vi.type.tensor_type.shape.CopyFrom(
                        vi_map[out_name].type.tensor_type.shape)
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[fp16_name], outputs=[out_name],
                    to=BFLOAT16, name=f"Cast_fp16_to_bf16_{out_name}",
                )
                node.output[i] = fp16_name
                new_nodes.append(cast_node)

    # Replace node list
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    return model


def export_embedding(model, output_dir: Path):
    """Export text embedding layer to ONNX."""
    print("Exporting embedding.onnx ...")
    wrapper = EmbedTokensWrapper(model).eval()

    input_ids = torch.randint(0, VOCAB_SIZE, (1, 32))

    output_path = output_dir / "embedding.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids,),
        str(output_path),
        opset_version=17,
        input_names=["input_ids"],
        output_names=["inputs_embeds"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "inputs_embeds": {0: "batch", 1: "seq_len"},
        },
    )
    _print_size(output_path)
    return output_path


def export_llm(model, output_dir: Path, dtype):
    """Export LLM decoder with dynamic KV cache to ONNX."""
    print("Exporting llm.onnx ...")
    wrapper = LLMWrapper(model).eval().to(dtype)

    batch = 1
    seq_len = 8
    past_len = 4

    inputs_embeds = torch.randn(batch, seq_len, HIDDEN_SIZE, dtype=dtype)
    attention_mask = torch.ones(batch, past_len + seq_len, dtype=torch.long)
    position_ids = (torch.arange(past_len, past_len + seq_len)
                    .view(1, 1, -1).expand(3, batch, -1).contiguous().long())

    past_kv = []
    for _ in range(NUM_LAYERS):
        past_kv.append(torch.randn(batch, NUM_KV_HEADS, past_len, HEAD_DIM, dtype=dtype))
        past_kv.append(torch.randn(batch, NUM_KV_HEADS, past_len, HEAD_DIM, dtype=dtype))

    # Build names and dynamic axes
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "inputs_embeds": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "total_seq_len"},
        "position_ids": {1: "batch", 2: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }

    for i in range(NUM_LAYERS):
        input_names.extend([f"past_key_{i}", f"past_value_{i}"])
        output_names.extend([f"present_key_{i}", f"present_value_{i}"])
        dynamic_axes[f"past_key_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"past_value_{i}"] = {0: "batch", 2: "past_seq_len"}
        dynamic_axes[f"present_key_{i}"] = {0: "batch", 2: "total_seq_len"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "total_seq_len"}

    output_path = output_dir / "llm.onnx"
    all_inputs = (inputs_embeds, attention_mask, position_ids) + tuple(past_kv)

    torch.onnx.export(
        wrapper,
        all_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    _print_size(output_path)
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────


def _print_size(onnx_path: Path):
    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    data_path = Path(str(onnx_path) + ".data")
    data_size = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
    total = onnx_size + data_size
    if data_size > 0:
        print(f"  {onnx_path.name}: {onnx_size:.1f} MB graph + {data_size:.1f} MB data = {total:.1f} MB")
    else:
        print(f"  {onnx_path.name}: {total:.1f} MB")


def validate_llm(wrapper, output_dir: Path, dtype):
    """Validate LLM ONNX output matches PyTorch."""
    print("\nValidating LLM ...")
    output_path = output_dir / "llm.onnx"
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(output_path), opts, providers=["CPUExecutionProvider"])

    print(f"  Prefill test (dtype={dtype}) ...", flush=True)
    embeds = torch.randn(1, 16, HIDDEN_SIZE, dtype=dtype)
    mask = torch.ones(1, 16, dtype=torch.long)
    pos = torch.arange(16).view(1, 1, -1).expand(3, 1, -1).contiguous().long()
    empty_kv = [torch.zeros(1, NUM_KV_HEADS, 0, HEAD_DIM, dtype=dtype)
                for _ in range(NUM_LAYERS * 2)]

    wrapper = wrapper.to(dtype)
    with torch.no_grad():
        pt_out = wrapper(embeds, mask, pos, *empty_kv)

    onnx_feed = {
        "inputs_embeds": embeds.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids": pos.numpy(),
    }
    for i in range(NUM_LAYERS):
        onnx_feed[f"past_key_{i}"] = empty_kv[i * 2].numpy()
        onnx_feed[f"past_value_{i}"] = empty_kv[i * 2 + 1].numpy()

    onnx_out = session.run(None, onnx_feed)
    diff = np.abs(pt_out[0].float().numpy() - onnx_out[0].astype(np.float32)).max()
    threshold = 1.0 if dtype == torch.float16 else 0.01
    print(f"    logits max diff: {diff:.2e} {'PASS' if diff < threshold else 'FAIL'}")


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Export GLM-OCR to ONNX")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    dtype_group = parser.add_mutually_exclusive_group()
    dtype_group.add_argument("--fp32", action="store_true",
                             help="Export in FP32 (debug)")
    dtype_group.add_argument("--bf16", action="store_true",
                             help="Export all models in BF16 (requires CUDA)")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip vision encoder export (use existing file)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine export dtype for embedding + LLM
    if args.fp32:
        export_dtype = torch.float32
    elif args.bf16:
        export_dtype = torch.bfloat16
    else:
        export_dtype = torch.float16

    print(f"Export dtype: {export_dtype}")
    print(f"Loading {MODEL_ID} (native BF16) ...")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)  # loads as BF16
    model.eval()
    if export_dtype != torch.bfloat16:
        model.to(export_dtype)  # BF16 -> FP16 or FP32
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    # Save processor/tokenizer/config for inference
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(str(output_dir))
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.save_pretrained(str(output_dir))
    print(f"Saved processor and config to {output_dir}\n")

    with torch.no_grad():
        export_embedding(model, output_dir)
        export_llm(model, output_dir, export_dtype)

    if args.skip_vision:
        print("Skipping vision encoder (--skip-vision)")
    else:
        if args.fp32:
            vision_precision = "fp32"
        elif args.bf16:
            vision_precision = "bf16"
        else:
            vision_precision = "fp16"
        export_vision_encoder(output_dir, precision=vision_precision)

    # Validate LLM (skip for BF16 — can't run on CPU)
    if args.bf16:
        print("\nSkipping LLM validation (BF16 requires CUDA)")
    else:
        llm_wrapper = LLMWrapper(model).eval()
        validate_llm(llm_wrapper, output_dir, export_dtype)

    # Summary
    print(f"\n{'='*50}")
    print("Export complete. Files:")
    total = 0
    for f in sorted(output_dir.glob("*.onnx*")):
        if f.suffix in ('.onnx', '.data'):
            sz = f.stat().st_size / 1024 / 1024
            total += sz
            print(f"  {f.name:40s} {sz:>8.1f} MB")
    print(f"  {'TOTAL':40s} {total:>8.1f} MB")
    if args.bf16:
        print("\nNote: BF16 models require CUDA. Conv ops in vision encoder stay FP16.")


if __name__ == "__main__":
    main()
