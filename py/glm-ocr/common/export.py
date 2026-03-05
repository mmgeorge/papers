"""Shared export wrappers and functions for GLM-OCR ONNX model export.

Used by both cuda/export.py and other/export.py.
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Transformers 5.2.0 monkey-patch ──────────────────────────────────────
#
# transformers 5.2.0 introduced a `video_processing_auto` module with a
# broken `video_processor_class_from_name()` function. It crashes during
# `AutoProcessor.from_pretrained()` when iterating VIDEO_PROCESSOR_MAPPING_NAMES,
# because it doesn't handle None entries or missing attributes.
#
# This patch provides a safe implementation that gracefully handles missing
# modules and None entries. Remove once transformers fixes the bug upstream.

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
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    DynamicCache,
)

MODEL_ID = "zai-org/GLM-OCR"
DEFAULT_OUTPUT_DIR = "../model"
DEFAULT_MAX_SEQ = 512

# Model architecture constants (from config.json → text_config).
# GLM-OCR uses Qwen2-based LLM with Grouped Query Attention (GQA):
#   - 16 decoder layers
#   - 12 attention heads, 8 KV heads (GQA ratio 3:2)
#   - head_dim = hidden_size / num_heads = 1536 / 12 = 128
#   - vocab_size = 59392 (including special tokens)
NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1536
VOCAB_SIZE = 59392


# ── Wrapper modules ──────────────────────────────────────────────────────


class EmbedTokensWrapper(nn.Module):
    """Wraps the text embedding layer for standalone ONNX export.

    Extracting embedding as a separate ONNX model lets the Rust runtime
    do token→embedding lookup independently from the LLM forward pass.
    This is used during prefill (embed all prompt tokens at once) and
    also by the non-CUDA-graph decode path (embed one token per step).

    The CUDA-graph decoder (llm_decoder.onnx, see DecoderStepWrapper)
    has the embedding table baked in, so it doesn't need this.
    """

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.language_model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLMWrapper(nn.Module):
    """Wraps the LLM decoder with explicit flattened KV cache I/O.

    HuggingFace's LLM expects a `DynamicCache` object for KV cache, but
    ONNX only supports flat tensor inputs/outputs. This wrapper:

    1. Takes 32 individual past_key/past_value tensors (16 layers × 2)
       as explicit named parameters
    2. Reconstructs a DynamicCache from them
    3. Calls the HuggingFace LLM
    4. Returns logits + 32 updated KV tensors as a flat tuple

    Why explicit named parameters instead of *args:
    torch.onnx.export in torch 2.10+ has issues mapping dynamic_axes to
    positional *args parameters. Using explicit named parameters
    (past_key_0, past_value_0, ...) ensures dynamic_axes dict keys
    match the actual parameter names reliably.

    The dynamic KV cache means past_seq_len grows by 1 each decode step.
    This is fine for the prefill pass (run once) but incompatible with
    CUDA graphs (which need fixed shapes). See DecoderStepWrapper for the
    fixed-shape alternative used in the decode loop.
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

        # Reconstruct HuggingFace's DynamicCache from flat tensors.
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

        # Flatten updated KV cache back to individual tensors for ONNX output.
        result = [logits]
        for layer in range(NUM_LAYERS):
            result.append(new_past.layers[layer].keys)
            result.append(new_past.layers[layer].values)
        return tuple(result)


class VisionEncoderWrapper(nn.Module):
    """Wraps the vision encoder for ONNX export with pre-computed position IDs.

    The vision encoder's rot_pos_emb() uses Python loops with data-dependent
    shapes (for t, h, w in grid_thw) that torch.onnx.export can't trace.
    This wrapper takes pre-computed pos_ids as input instead, avoiding all
    data-dependent control flow.

    For single-image inference, attention is full (no cu_seqlens chunking).
    The eager attention path in GlmOcrVisionAttention splits by cu_seqlens
    using torch.split(lengths.tolist()), which is also untraceable. We bypass
    this by re-implementing attention inline with standard matmul attention.

    Inputs:
        pixel_values:  [num_patches, 1176] bf16 — flattened image patches
        pos_ids:       [num_patches, 2] int64 — pre-computed (h, w) position pairs
        max_grid_size: [] int64 scalar — max(h, w) for RoPE frequency table

    Output:
        hidden_states: [num_merged_patches, 1536] bf16 — vision embeddings
    """

    def __init__(self, visual_model):
        super().__init__()
        self.patch_embed = visual_model.patch_embed
        self.rotary_pos_emb = visual_model.rotary_pos_emb
        self.blocks = visual_model.blocks
        self.post_layernorm = visual_model.post_layernorm
        self.downsample = visual_model.downsample
        self.merger = visual_model.merger

        # Copy attention config from first block for inline attention
        attn = visual_model.blocks[0].attn
        self._num_heads = attn.num_heads
        self._head_dim = attn.head_dim
        self._scaling = attn.scaling

    def _attention_forward(self, attn_module, hidden_states, position_embeddings):
        """Inline attention: QKV projection → RMSNorm → RoPE → matmul attention.

        Replaces GlmOcrVisionAttention.forward() to avoid the untraceable
        cu_seqlens splitting path. Uses standard scaled dot-product attention
        over the full sequence (single-image, no chunking needed).
        """
        seq_len = hidden_states.shape[0]

        # QKV projection + reshape to [Q, K, V] each [seq, heads, head_dim]
        qkv = attn_module.qkv(hidden_states)
        q, k, v = qkv.reshape(seq_len, 3, self._num_heads, self._head_dim).permute(1, 0, 2, 3).unbind(0)

        # Per-head RMSNorm (applied before RoPE, unusual for this architecture)
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)

        # 2D RoPE
        from transformers.models.glm_ocr.modeling_glm_ocr import apply_rotary_pos_emb_vision
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Reshape for batched matmul: [seq, heads, dim] → [1, heads, seq, dim]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # Standard scaled dot-product attention (no mask — bidirectional)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self._scaling
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [1, heads, seq, dim] → [seq, hidden]
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_len, -1).contiguous()
        return attn_module.proj(attn_output)

    def forward(self, pixel_values, pos_ids, max_grid_size):
        hidden_states = self.patch_embed(pixel_values)

        # Compute RoPE from pre-computed pos_ids (no data-dependent shapes)
        freqs = self.rotary_pos_emb(max_grid_size)  # [max_grid, head_dim//2]
        rope = freqs[pos_ids].flatten(1)             # [N, head_dim]
        emb = torch.cat((rope, rope), dim=-1)        # [N, head_dim*2]
        position_embeddings = (emb.cos(), emb.sin())

        # Run 24 blocks with inline attention (bypasses cu_seqlens splitting)
        for blk in self.blocks:
            hidden_states = hidden_states + self._attention_forward(
                blk.attn, blk.norm1(hidden_states), position_embeddings,
            )
            hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))

        # Post-layernorm + spatial downsample (2×2 merge via Conv2d)
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.view(-1, 2, 2, hidden_states.shape[-1])
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.downsample.out_channels)

        # Patch merger (gated MLP)
        return self.merger(hidden_states)


# ── Export functions ──────────────────────────────────────────────────────


def export_vision_encoder(model, output_dir: Path, dtype, apply_mha=True):
    """Export the vision encoder to ONNX.

    Pipeline: wrap visual model → trace in FP32 → optional MHA optimization →
    optional FP32→BF16 conversion.

    Args:
        model: The HuggingFace model.
        output_dir: Directory for output files.
        dtype: torch.bfloat16 for BF16 conversion, torch.float32 for FP32.
        apply_mha: If True, apply MHA surgery (CUDA). If False, skip (CPU/CoreML).
    """
    if apply_mha:
        output_name = "vision_encoder_mha.onnx"
    else:
        output_name = "vision_encoder.onnx"

    print(f"Exporting {output_name} ...")
    # Trace in FP32 for clean graph structure (MHA surgery needs FP32 patterns)
    wrapper = VisionEncoderWrapper(model.model.visual).eval().to(torch.float32)

    # Example inputs: 24×24 grid = 576 patches (standard 336×336 image)
    num_patches = 576
    pixel_values = torch.randn(num_patches, 1176, dtype=torch.float32)
    pos_ids = torch.zeros(num_patches, 2, dtype=torch.long)
    max_grid_size = torch.tensor(24, dtype=torch.long)

    # Fill pos_ids with realistic values for tracing
    idx = 0
    for bh in range(12):
        for bw in range(12):
            for mh in range(2):
                for mw in range(2):
                    pos_ids[idx, 0] = bh * 2 + mh
                    pos_ids[idx, 1] = bw * 2 + mw
                    idx += 1

    final_path = output_dir / output_name

    if apply_mha:
        # Export to temp file first — MHA surgery reads raw and writes final
        raw_path = output_dir / "vision_encoder_raw.onnx"
        torch.onnx.export(
            wrapper,
            (pixel_values, pos_ids, max_grid_size),
            str(raw_path),
            opset_version=17,
            input_names=["pixel_values", "pos_ids", "max_grid_size"],
            output_names=["hidden_states"],
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "pos_ids": {0: "num_patches"},
                "hidden_states": {0: "num_merged_patches"},
            },
        )
        _print_size(raw_path)

        from optimize_mha_vision import apply_mha_surgery
        print("  Applying MHA surgery ...")
        apply_mha_surgery(raw_path, final_path)
        raw_path.unlink()
        raw_data = raw_path.with_suffix(".onnx.data")
        if raw_data.exists():
            raw_data.unlink()
    else:
        # Export directly to final path (external data filename matches)
        torch.onnx.export(
            wrapper,
            (pixel_values, pos_ids, max_grid_size),
            str(final_path),
            opset_version=17,
            input_names=["pixel_values", "pos_ids", "max_grid_size"],
            output_names=["hidden_states"],
            dynamic_axes={
                "pixel_values": {0: "num_patches"},
                "pos_ids": {0: "num_patches"},
                "hidden_states": {0: "num_merged_patches"},
            },
        )
        _print_size(final_path)

    # Convert FP32→BF16 (direct, no FP16 intermediate)
    if dtype == torch.bfloat16:
        from common.onnx_utils import convert_fp32_to_bf16
        print("  Converting FP32 → BF16 ...")
        convert_fp32_to_bf16(final_path)

    _print_size(final_path)
    return final_path


def export_embedding(model, output_dir: Path):
    """Export the token embedding table as a standalone ONNX model.

    Input:  input_ids    [batch, seq_len] int64
    Output: inputs_embeds [batch, seq_len, 1536] bf16

    Both batch and seq_len are dynamic axes, so this works for both
    prefill (seq_len = prompt length) and single-token decode steps.
    """
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
    """Export the LLM decoder with dynamic KV cache to ONNX.

    This model is used for the **prefill pass** — a single forward pass
    with the full prompt (text tokens + vision embeddings merged in).

    The KV cache uses dynamic past_seq_len (starts at 0 for prefill,
    grows by 1 each decode step). This makes it incompatible with CUDA
    graphs, which require fixed shapes. See export_decoder() and
    DecoderStepWrapper for the CUDA-graph-compatible alternative.
    """
    print("Exporting llm.onnx ...")
    wrapper = LLMWrapper(model).eval().to(dtype)

    batch = 1
    seq_len = 8
    past_len = 4

    inputs_embeds = torch.randn(batch, seq_len, HIDDEN_SIZE, dtype=dtype)
    attention_mask = torch.ones(batch, past_len + seq_len, dtype=torch.long)
    # M-RoPE: 3D position IDs [3, batch, seq_len] — temporal, height, width
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


def load_model(dtype, output_dir: Path):
    """Load the HuggingFace model and save processor/config to output_dir.

    Returns (model, processor, config).
    """
    print(f"Loading {MODEL_ID} (native BF16) ...")
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID)  # loads as BF16
    model.eval()
    if dtype != torch.bfloat16:
        model.to(dtype)  # BF16 -> FP16 or FP32
    print(f"Loaded in {time.perf_counter() - t0:.1f}s\n")

    # Save processor/tokenizer/config for inference.
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(str(output_dir))
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.save_pretrained(str(output_dir))
    print(f"Saved processor and config to {output_dir}\n")

    return model, processor, config
