"""Shared FP32 ONNX export logic for PP-FormulaNet encoder and decoder."""

import os
import sys
import warnings

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import onnx

from .encoder import load_encoder
from .decoder import (
    load_decoder, PPFormulaNetDecoder,
    N_HEADS, HEAD_DIM, N_LAYERS, ENCODER_DIM, MAX_SEQ,
)

ENC_SEQ_LEN = 144  # encoder output tokens for 768x768 input


class DecoderForExport(torch.nn.Module):
    """Wrapper with flat fixed-size self-attn KV cache I/O for ONNX export.

    Cross-attention KV is always recomputed from encoder_hidden_states,
    so only self-attention KV (16 tensors) crosses the ONNX boundary.
    All KV tensors have static shape [B, 16, MAX_SEQ, 32].
    """

    def __init__(self, decoder: PPFormulaNetDecoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids, encoder_hidden_states, step, *past_self_kv_flat):
        """
        Args:
            input_ids: [B, 1]
            encoder_hidden_states: [B, enc_seq, 1024]
            step: [1] int64
            past_self_kv_flat: 16 tensors (2 per layer):
                self_attn_key   [B, 16, MAX_SEQ, 32]
                self_attn_value [B, 16, MAX_SEQ, 32]

        Returns:
            logits: [B, 1, 50000]
            16 present self-attn KV tensors (same fixed shapes)
        """
        past_key_values = []
        for layer in range(N_LAYERS):
            base = layer * 2
            past_key_values.append((
                past_self_kv_flat[base],
                past_self_kv_flat[base + 1],
                None,
                None,
            ))

        logits, present_key_values = self.decoder(
            input_ids, encoder_hidden_states, past_key_values, step
        )

        present_flat = []
        for layer_kv in present_key_values:
            present_flat.append(layer_kv[0])
            present_flat.append(layer_kv[1])

        return (logits, *present_flat)


def _build_input_names():
    names = ["input_ids", "encoder_hidden_states", "step"]
    for layer in range(N_LAYERS):
        names.append(f"past_key_values.{layer}.key")
        names.append(f"past_key_values.{layer}.value")
    return names


def _build_output_names():
    names = ["logits"]
    for layer in range(N_LAYERS):
        names.append(f"present_key_values.{layer}.key")
        names.append(f"present_key_values.{layer}.value")
    return names


def _build_dynamic_axes():
    axes = {
        "input_ids": {0: "batch"},
        "encoder_hidden_states": {0: "batch", 1: "encoder_seq"},
        "logits": {0: "batch"},
    }
    for layer in range(N_LAYERS):
        for kv in ["key", "value"]:
            axes[f"past_key_values.{layer}.{kv}"] = {0: "batch"}
            axes[f"present_key_values.{layer}.{kv}"] = {0: "batch"}
    return axes


def export_encoder_fp32(weights_dir, output_dir):
    """Export PyTorch encoder to FP32 ONNX.

    Args:
        weights_dir: Directory containing encoder_weights.npz.
        output_dir: Directory for the output encoder.onnx.

    Returns:
        Path to the saved ONNX model.
    """
    weights_path = os.path.join(weights_dir, "encoder_weights.npz")
    output_path = os.path.join(output_dir, "encoder.onnx")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading PyTorch encoder...")
    model = load_encoder(weights_path)

    dummy_input = torch.randn(1, 1, 768, 768)

    print(f"Exporting to {output_path}...")
    warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["pixel_values"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print("Validating...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done: {output_path} ({size_mb:.1f} MB, {len(onnx_model.graph.node)} nodes)")
    return output_path


def export_decoder_fp32(weights_dir, output_dir):
    """Export PyTorch decoder to FP32 ONNX with fixed-size KV cache.

    Args:
        weights_dir: Directory containing decoder_weights.npz.
        output_dir: Directory for the output decoder.onnx.

    Returns:
        Path to the saved ONNX model.
    """
    weights_path = os.path.join(weights_dir, "decoder_weights.npz")
    output_path = os.path.join(output_dir, "decoder.onnx")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading PyTorch decoder...")
    decoder = load_decoder(weights_path)
    model = DecoderForExport(decoder)
    model.eval()

    batch = 1
    dummy_input_ids = torch.zeros(batch, 1, dtype=torch.long)
    dummy_encoder = torch.randn(batch, ENC_SEQ_LEN, ENCODER_DIM)
    dummy_step = torch.zeros(1, dtype=torch.long)

    dummy_past_kv = []
    for _ in range(N_LAYERS):
        dummy_past_kv.append(torch.zeros(batch, N_HEADS, MAX_SEQ, HEAD_DIM))
        dummy_past_kv.append(torch.zeros(batch, N_HEADS, MAX_SEQ, HEAD_DIM))

    dummy_inputs = (dummy_input_ids, dummy_encoder, dummy_step, *dummy_past_kv)

    input_names = _build_input_names()
    output_names = _build_output_names()
    dynamic_axes = _build_dynamic_axes()

    print(f"Exporting to {output_path}...")
    print(f"  Inputs: {len(input_names)} (3 fixed + {N_LAYERS * 2} self-attn KV)")
    print(f"  Outputs: {len(output_names)} (1 logits + {N_LAYERS * 2} self-attn KV)")
    print(f"  KV buffer shape: [B, {N_HEADS}, {MAX_SEQ}, {HEAD_DIM}] (fixed)")

    warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX export.*")
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print("Validating...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done: {output_path} ({size_mb:.1f} MB, {len(onnx_model.graph.node)} nodes)")
    return output_path
