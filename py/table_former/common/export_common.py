"""ONNX export wrappers and utility functions."""

import warnings
from pathlib import Path

import torch
import torch.nn as nn

from .decoder import MAX_SEQ
from .weights import D_MODEL, N_HEADS, HEAD_DIM, N_LAYERS, VOCAB_SIZE
from .encoder import NUM_POSITIONS


def export_encoder(model: nn.Module, output_path: Path):
    """Export encoder to ONNX with fully static shapes."""
    model.eval()
    dummy = torch.randn(1, 3, 448, 448)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            input_names=["pixel_values"],
            output_names=["encoder_memory", "enc_out_raw"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  encoder.onnx: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def export_decoder(model: nn.Module, output_path: Path):
    """Export decoder to ONNX with fully static shapes (CUDA-graph-friendly)."""
    model.eval()

    # Build input names
    input_names = ["input_ids", "encoder_memory", "step"]
    for layer in range(N_LAYERS):
        input_names.append(f"past_key_values.{layer}.key")
        input_names.append(f"past_key_values.{layer}.value")

    # Build output names
    output_names = ["logits", "hidden_state"]
    for layer in range(N_LAYERS):
        output_names.append(f"present_key_values.{layer}.key")
        output_names.append(f"present_key_values.{layer}.value")

    # Dummy inputs — all static shapes
    dummy_input_ids = torch.zeros(1, 1, dtype=torch.long)
    dummy_encoder = torch.randn(1, NUM_POSITIONS, D_MODEL)
    dummy_step = torch.zeros(1, dtype=torch.long)
    dummy_kv = []
    for _ in range(N_LAYERS):
        dummy_kv.append(torch.zeros(1, N_HEADS, MAX_SEQ, HEAD_DIM))  # key
        dummy_kv.append(torch.zeros(1, N_HEADS, MAX_SEQ, HEAD_DIM))  # value

    dummy_inputs = (dummy_input_ids, dummy_encoder, dummy_step, *dummy_kv)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            model,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  decoder.onnx: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def export_bbox_decoder(model: nn.Module, output_path: Path):
    """Export bbox decoder to ONNX with dynamic N cells axis."""
    model.eval()

    dummy_enc = torch.randn(1, 256, 28, 28)
    dummy_cells = torch.randn(10, D_MODEL)  # 10 cells as example

    dynamic_axes = {
        "cell_hidden_states": {0: "num_cells"},
        "bboxes": {0: "num_cells"},
        "classes": {0: "num_cells"},
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        torch.onnx.export(
            model,
            (dummy_enc, dummy_cells),
            str(output_path),
            input_names=["enc_out_raw", "cell_hidden_states"],
            output_names=["bboxes", "classes"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  bbox_decoder.onnx: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
