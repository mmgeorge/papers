"""Export a CUDA-graph-friendly decode-step LLM for GLM-OCR.

The existing llm.onnx uses dynamic KV cache shapes (past_seq_len grows each
step). CUDA graphs require all tensor shapes to be fixed. This script exports
a decode-only LLM with:

  1. Embedded token lookup (input_ids [1,1] → internal embedding, no separate
     embedding session call per step)
  2. Fixed-size KV cache: past_key_{i} [1, 8, MAX_SEQ, 128] (not dynamic)
     Uses StaticCache + cache_position for indexed writes (index_copy_)
  3. Step counter + prefill_len inputs → derive cache_position, attention_mask,
     and 3D M-RoPE position_ids internally
  4. ArgMax built into the graph (next_token [1] stays on GPU)
  5. In-place KV update: present outputs share the same buffer as past inputs

The prefill pass still uses llm.onnx with dynamic shapes (runs once per
formula, no CUDA graph needed).

Usage:
    python export_decoder.py [--max-seq 512] [--output-dir ./model]
    python export_decoder.py --validate  # run numerical validation
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
from onnx import helper, TensorProto
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxconverter_common import float16
from transformers import AutoModelForImageTextToText, StaticCache
from transformers.cache_utils import StaticLayer

MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_OUTPUT_DIR = "../model"
DEFAULT_MAX_SEQ = 512

# From config.json
NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1536
VOCAB_SIZE = 59392


class WhereStaticLayer(StaticLayer):
    """StaticLayer that uses torch.where instead of index_copy_ for updates.

    index_copy_ exports to ONNX ScatterND which is incompatible with CUDA graph
    capture in ORT. torch.where exports to ONNX Where + comparison ops, which
    are fully CUDA-graph compatible.
    """

    def update(self, key_states, value_states, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        cache_position = (
            cache_position if cache_position is not None else torch.arange(key_states.shape[-2], device=self.device)
        )
        # Build a boolean mask: [1, 1, MAX_SEQ, 1] — True only at cache_position
        positions = torch.arange(self.keys.shape[2], device=self.keys.device)
        mask = (positions == cache_position.squeeze()).view(1, 1, -1, 1)
        # Use Where to do the update: replace at cache_position, keep rest
        self.keys = torch.where(mask, key_states.expand_as(self.keys), self.keys)
        self.values = torch.where(mask, value_states.expand_as(self.values), self.values)
        return self.keys, self.values


class WhereStaticCache(StaticCache):
    """StaticCache using WhereStaticLayer for CUDA-graph-compatible updates."""

    def __init__(self, config, max_cache_len):
        # Call parent init which creates StaticLayer instances
        super().__init__(config, max_cache_len=max_cache_len)
        # Replace all layers with our Where-based variant
        for i, layer in enumerate(self.layers):
            new_layer = WhereStaticLayer(max_cache_len=max_cache_len)
            # Copy attributes from the original (already initialized or not)
            if layer.is_initialized:
                new_layer.keys = layer.keys
                new_layer.values = layer.values
                new_layer.is_initialized = True
                new_layer.dtype = layer.dtype
                new_layer.device = layer.device
                new_layer.max_batch_size = layer.max_batch_size
                new_layer.num_heads = layer.num_heads
                new_layer.k_head_dim = layer.k_head_dim
                new_layer.v_head_dim = layer.v_head_dim
            self.layers[i] = new_layer


class DecoderStepWrapper(nn.Module):
    """Wraps the GLM-OCR LLM for a single decode step with fixed-size KV cache.

    Uses a custom WhereStaticCache that replaces index_copy_ (ScatterND) with
    torch.where (Where op), making the exported ONNX graph fully compatible
    with CUDA graph capture.

    Inputs:
        input_ids: [1, 1] int64 — single token to decode
        step: [1] int64 — decode step (0 = first generated token)
        prefill_len: [1] int64 — length of the prefill KV cache
        past_key_{i}: [1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM] × NUM_LAYERS
        past_value_{i}: [1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM] × NUM_LAYERS

    Outputs:
        next_token: [1] int64 — argmax of logits
        present_key_{i}: [1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM] × NUM_LAYERS
        present_value_{i}: [1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM] × NUM_LAYERS
    """

    def __init__(self, model, max_seq: int):
        super().__init__()
        self.embed_tokens = model.model.language_model.embed_tokens
        self.language_model = model.model.language_model
        self.lm_head = model.lm_head
        self.max_seq = max_seq
        self.config = model.config.text_config

    def forward(self, input_ids, step, prefill_len,
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

        # Embed the single input token
        inputs_embeds = self.embed_tokens(input_ids)  # [1, 1, HIDDEN_SIZE]

        # Compute write position in the KV cache
        cache_pos = prefill_len + step  # [1] — where to write new KV entry

        # Build 4D causal attention mask directly, bypassing create_causal_mask()
        # which generates a GatherND op that breaks CUDA graph capture.
        # Shape: [1, 1, 1, MAX_SEQ] — 0.0 for attended, min_dtype for masked
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        positions = torch.arange(self.max_seq, device=input_ids.device)
        attend = positions < (cache_pos + 1)  # [MAX_SEQ] bool
        attention_mask = torch.where(
            attend.view(1, 1, 1, -1),
            torch.tensor(0.0, dtype=inputs_embeds.dtype, device=input_ids.device),
            torch.tensor(min_dtype, dtype=inputs_embeds.dtype, device=input_ids.device),
        )

        # Build 3D M-RoPE position IDs for decode step
        # During decode, all 3 dims get the same position
        position_ids = cache_pos.view(1, 1, 1).expand(3, 1, 1).contiguous()

        # Build WhereStaticCache (uses Where op instead of ScatterND)
        past_key_values = WhereStaticCache(
            config=self.config,
            max_cache_len=self.max_seq,
        )

        # Inject the input KV buffers into the cache layers
        for layer_idx in range(NUM_LAYERS):
            layer = past_key_values.layers[layer_idx]
            layer.keys = past_keys[layer_idx]
            layer.values = past_values[layer_idx]
            layer.is_initialized = True
            layer.dtype = past_keys[layer_idx].dtype
            layer.device = past_keys[layer_idx].device
            layer.max_batch_size = 1
            layer.num_heads = NUM_KV_HEADS
            layer.k_head_dim = HEAD_DIM
            layer.v_head_dim = HEAD_DIM

        # Run the language model with WhereStaticCache + cache_position
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_pos,
        )

        logits = self.lm_head(outputs.last_hidden_state)  # [1, 1, VOCAB_SIZE]

        # ArgMax for next token
        next_token = logits[:, -1, :].argmax(dim=-1)  # [1]

        # Extract updated KV cache
        updated_cache = outputs.past_key_values
        result = [next_token]
        for layer_idx in range(NUM_LAYERS):
            result.append(updated_cache.layers[layer_idx].keys)
            result.append(updated_cache.layers[layer_idx].values)

        return tuple(result)


def topological_sort(graph):
    """Fix node ordering after float16 conversion inserts Cast nodes."""
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


def convert_fp16(input_path, output_path):
    """Convert FP32 ONNX to FP16 with native I/O types."""
    print(f"Converting to FP16: {input_path} -> {output_path}")
    model = onnx.load(str(input_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=False,
        )

    topological_sort(model_fp16.graph)
    onnx.save(model_fp16, str(output_path))
    print(f"  FP16 model saved: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


def export_decoder(model, output_dir: Path, max_seq: int):
    """Export the CUDA-graph-friendly decode-step LLM."""
    print(f"Exporting llm_decoder.onnx (MAX_SEQ={max_seq}) ...")

    # Model is already FP16 from main(), so .half() is technically redundant
    # but makes the intent clear. The export produces FP16 weights directly.
    wrapper = DecoderStepWrapper(model, max_seq).eval().half()

    batch = 1
    seq = 1

    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq), dtype=torch.long)
    step = torch.tensor([0], dtype=torch.long)
    prefill_len = torch.tensor([10], dtype=torch.long)

    past_kv = []
    for _ in range(NUM_LAYERS):
        past_kv.append(torch.randn(batch, NUM_KV_HEADS, max_seq, HEAD_DIM, dtype=torch.float16))
        past_kv.append(torch.randn(batch, NUM_KV_HEADS, max_seq, HEAD_DIM, dtype=torch.float16))

    input_names = ["input_ids", "step", "prefill_len"]
    output_names = ["next_token"]

    for i in range(NUM_LAYERS):
        input_names.extend([f"past_key_{i}", f"past_value_{i}"])
        output_names.extend([f"present_key_{i}", f"present_value_{i}"])

    output_path = output_dir / "llm_decoder.onnx"
    all_inputs = (input_ids, step, prefill_len) + tuple(past_kv)

    torch.onnx.export(
        wrapper,
        all_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
    )

    # Model weights are already FP16 — no onnxconverter conversion needed.
    _print_size(output_path)
    return output_path


def validate_decoder(output_dir: Path, max_seq: int):
    """Validate the exported decoder ONNX."""
    print("\nValidating llm_decoder.onnx ...")
    decoder_path = output_dir / "llm_decoder.onnx"

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(decoder_path), opts, providers=["CPUExecutionProvider"])

    print("  Session inputs:")
    for inp in session.get_inputs():
        print(f"    {inp.name}: {inp.type} {inp.shape}")
    print("  Session outputs:")
    for out in session.get_outputs():
        print(f"    {out.name}: {out.type} {out.shape}")

    # Test inference
    feed = {
        "input_ids": np.array([[100]], dtype=np.int64),
        "step": np.array([0], dtype=np.int64),
        "prefill_len": np.array([10], dtype=np.int64),
    }
    for i in range(NUM_LAYERS):
        feed[f"past_key_{i}"] = np.zeros(
            (1, NUM_KV_HEADS, max_seq, HEAD_DIM), dtype=np.float16
        )
        feed[f"past_value_{i}"] = np.zeros(
            (1, NUM_KV_HEADS, max_seq, HEAD_DIM), dtype=np.float16
        )

    outputs = session.run(None, feed)
    next_token = outputs[0]
    print(f"  next_token: {next_token} (shape={next_token.shape}, dtype={next_token.dtype})")

    # Check KV outputs have expected shapes
    for i in range(NUM_LAYERS):
        k = outputs[1 + i * 2]
        v = outputs[2 + i * 2]
        assert k.shape == (1, NUM_KV_HEADS, max_seq, HEAD_DIM), \
            f"present_key_{i} shape mismatch: {k.shape}"
        assert v.shape == (1, NUM_KV_HEADS, max_seq, HEAD_DIM), \
            f"present_value_{i} shape mismatch: {v.shape}"

    print("  PASS: all shapes correct")


def _print_size(onnx_path: Path):
    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    data_path = Path(str(onnx_path) + ".data")
    data_size = data_path.stat().st_size / 1024 / 1024 if data_path.exists() else 0
    total = onnx_size + data_size
    if data_size > 0:
        print(f"  {onnx_path.name}: {onnx_size:.1f} MB graph + {data_size:.1f} MB data = {total:.1f} MB")
    else:
        print(f"  {onnx_path.name}: {total:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export CUDA-graph-friendly decode-step LLM for GLM-OCR"
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ,
                        help=f"Maximum sequence length for fixed KV cache (default: {DEFAULT_MAX_SEQ})")
    parser.add_argument("--validate", action="store_true",
                        help="Run numerical validation after export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)
    model.eval()
    model.to(torch.float16)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    with torch.no_grad():
        export_decoder(model, output_dir, args.max_seq)

    if args.validate:
        validate_decoder(output_dir, args.max_seq)

    # Summary
    decoder_path = output_dir / "llm_decoder.onnx"
    print(f"\n{'='*50}")
    print(f"Export complete: {decoder_path}")
    _print_size(decoder_path)


if __name__ == "__main__":
    main()
