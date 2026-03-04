"""Shared helpers for GLM-OCR ONNX inference.

Constants, dtype detection, device resolution, and M-RoPE position ID computation.
"""

import importlib
import os

import numpy as np
import onnxruntime as ort
from ml_dtypes import bfloat16 as np_bfloat16

# ── Model constants from config.json ─────────────────────────────────

NUM_LAYERS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1536
SPATIAL_MERGE_SIZE = 2

os.environ["OMP_NUM_THREADS"] = "1"


# ── Transformers monkey-patch ────────────────────────────────────────
# transformers 5.2.0 tries to load a VideoProcessor for Qwen2-VL-based
# models (including GLM-OCR), which requires PyTorch/torchvision. We don't
# need video processing, so patch out the video processor loading and the
# type check that rejects None.

def _apply_transformers_patch():
    try:
        from transformers.models.auto.video_processing_auto import AutoVideoProcessor
        from transformers import processing_utils
    except ImportError:
        return

    # Return None from AutoVideoProcessor.from_pretrained
    AutoVideoProcessor.from_pretrained = classmethod(lambda cls, *a, **kw: None)

    # Allow None for video_processor in ProcessorMixin type check
    _original_check = processing_utils.ProcessorMixin.check_argument_for_proper_class

    def _lenient_check(self, argument_name, argument):
        if argument is None and "video" in argument_name:
            return None
        return _original_check(self, argument_name, argument)

    processing_utils.ProcessorMixin.check_argument_for_proper_class = _lenient_check


_apply_transformers_patch()


# ── Helpers ──────────────────────────────────────────────────────────

def detect_dtype(session):
    """Detect numpy float dtype from an ORT session's first float input."""
    input_type = session.get_inputs()[0].type
    if "bfloat16" in input_type:
        return np_bfloat16
    elif "float16" in input_type:
        return np.float16
    return np.float32


def resolve_device(device: str):
    """Resolve device string into (providers, session_options)."""
    available = ort.get_available_providers()

    if device == "auto":
        if "CUDAExecutionProvider" in available:
            device = "cuda"
        elif "DmlExecutionProvider" in available:
            device = "dml"
        else:
            device = "cpu"

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider not available. Available: {available}")
        providers = [
            ("CUDAExecutionProvider", {
                "use_tf32": 1,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "cudnn_conv_use_max_workspace": "1",
                "arena_extend_strategy": "kSameAsRequested",
            }),
            "CPUExecutionProvider",
        ]
    elif device == "dml":
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(f"DmlExecutionProvider not available. Available: {available}")
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = False
    elif device == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"Unknown device: {device!r}")

    print(f"Using device: {device} (providers: {[p if isinstance(p, str) else p[0] for p in providers]})")
    return providers, opts


def compute_vision_pos_ids(grid_thw):
    """Compute position IDs for vision rotary embeddings (M-RoPE)."""
    pos_ids_list = []
    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)

        hpos_ids = np.arange(h).reshape(-1, 1).repeat(w, axis=1)
        hpos_ids = hpos_ids.reshape(
            h // SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE,
            w // SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3).flatten()

        wpos_ids = np.arange(w).reshape(1, -1).repeat(h, axis=0)
        wpos_ids = wpos_ids.reshape(
            h // SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE,
            w // SPATIAL_MERGE_SIZE, SPATIAL_MERGE_SIZE,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3).flatten()

        frame_pos = np.stack([hpos_ids, wpos_ids], axis=-1)
        pos_ids_list.append(np.tile(frame_pos, (t, 1)))

    pos_ids = np.concatenate(pos_ids_list, axis=0)
    max_grid_size = max(max(int(h), int(w)) for _, h, w in grid_thw)
    return pos_ids, max_grid_size


def build_position_ids(input_ids, image_grid_thw, image_token_id):
    """Build 3D M-RoPE position IDs for text+vision tokens.

    Returns [3, batch, seq_len] position IDs where:
    - dim 0: temporal position (constant per frame for vision tokens)
    - dim 1: height/row position (spatial for vision tokens)
    - dim 2: width/col position (spatial for vision tokens)
    Text tokens get the same value in all 3 dims.
    """
    batch_size, seq_len = input_ids.shape
    position_ids = np.zeros((3, batch_size, seq_len), dtype=np.int64)

    for b in range(batch_size):
        ids = input_ids[b]
        image_mask = (ids == image_token_id)
        pos = 0
        i = 0
        img_idx = 0

        while i < seq_len:
            if not image_mask[i]:
                position_ids[0, b, i] = pos
                position_ids[1, b, i] = pos
                position_ids[2, b, i] = pos
                pos += 1
                i += 1
            else:
                t_grid = int(image_grid_thw[img_idx, 0])
                h_grid = int(image_grid_thw[img_idx, 1])
                w_grid = int(image_grid_thw[img_idx, 2])
                merged_h = h_grid // SPATIAL_MERGE_SIZE
                merged_w = w_grid // SPATIAL_MERGE_SIZE
                num_vision_tokens = t_grid * merged_h * merged_w

                temporal_pos = pos
                for vi in range(num_vision_tokens):
                    row = (vi % (merged_h * merged_w)) // merged_w
                    col = (vi % (merged_h * merged_w)) % merged_w
                    position_ids[0, b, i + vi] = temporal_pos
                    position_ids[1, b, i + vi] = pos + row
                    position_ids[2, b, i + vi] = pos + col

                pos = pos + max(merged_h, merged_w)
                i += num_vision_tokens
                img_idx += 1

    return position_ids
