"""Load TableFormer weights from safetensors checkpoint."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


D_MODEL = 512
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 64
N_LAYERS = 6
VOCAB_SIZE = 13


def download_weights() -> Path:
    """Download docling-models and return path to accurate safetensors."""
    dl = snapshot_download(repo_id="ds4sd/docling-models")
    return (
        Path(dl)
        / "model_artifacts"
        / "tableformer"
        / "accurate"
        / "tableformer_accurate.safetensors"
    )


def _split_in_proj(weight, bias):
    """Split nn.MultiheadAttention in_proj_weight [3*D, D] into Q, K, V."""
    d = D_MODEL
    q_w, k_w, v_w = weight[:d], weight[d : 2 * d], weight[2 * d :]
    q_b, k_b, v_b = bias[:d], bias[d : 2 * d], bias[2 * d :]
    return (q_w, q_b), (k_w, k_b), (v_w, v_b)


def load_weights(safetensors_path: Path):
    """Load and remap weights into encoder, decoder, and bbox_decoder state dicts."""
    encoder_sd = {}
    decoder_sd = {}
    bbox_sd = {}

    with safe_open(safetensors_path, framework="pt") as f:
        all_tensors = {name: f.get_tensor(name) for name in f.keys()}

    for name, tensor in all_tensors.items():
        # --- Encoder: ResNet backbone ---
        if name.startswith("_encoder._resnet."):
            new_name = name.replace("_encoder._resnet.", "resnet.")
            encoder_sd[new_name] = tensor

        # --- Encoder: input_filter (resnet_block 256→512) ---
        elif name.startswith("_tag_transformer._input_filter."):
            new_name = name.replace(
                "_tag_transformer._input_filter.", "input_filter."
            )
            encoder_sd[new_name] = tensor

        # --- Encoder: TransformerEncoder ---
        elif name.startswith("_tag_transformer._encoder."):
            new_name = name.replace(
                "_tag_transformer._encoder.", "transformer_encoder."
            )
            encoder_sd[new_name] = tensor

        # --- Decoder: embedding ---
        elif name.startswith("_tag_transformer._embedding."):
            new_name = name.replace("_tag_transformer._embedding.", "embedding.")
            decoder_sd[new_name] = tensor

        # --- Decoder: positional encoding buffer ---
        elif name == "_tag_transformer._positional_encoding.pe":
            decoder_sd["pe"] = tensor

        # --- Decoder: output FC ---
        elif name.startswith("_tag_transformer._fc."):
            new_name = name.replace("_tag_transformer._fc.", "fc.")
            decoder_sd[new_name] = tensor

        # --- Decoder: transformer decoder layers ---
        elif name.startswith("_tag_transformer._decoder.layers."):
            rest = name.replace("_tag_transformer._decoder.layers.", "")
            # rest is like "0.self_attn.in_proj_weight"
            layer_idx = int(rest.split(".")[0])
            param_path = ".".join(rest.split(".")[1:])
            prefix = f"layers.{layer_idx}."

            if param_path == "self_attn.in_proj_weight":
                bias = all_tensors[
                    f"_tag_transformer._decoder.layers.{layer_idx}.self_attn.in_proj_bias"
                ]
                (q_w, q_b), (k_w, k_b), (v_w, v_b) = _split_in_proj(
                    tensor, bias
                )
                decoder_sd[prefix + "self_attn.q_proj.weight"] = q_w
                decoder_sd[prefix + "self_attn.q_proj.bias"] = q_b
                decoder_sd[prefix + "self_attn.k_proj.weight"] = k_w
                decoder_sd[prefix + "self_attn.k_proj.bias"] = k_b
                decoder_sd[prefix + "self_attn.v_proj.weight"] = v_w
                decoder_sd[prefix + "self_attn.v_proj.bias"] = v_b
            elif param_path == "self_attn.in_proj_bias":
                pass  # handled above with weight
            elif param_path == "multihead_attn.in_proj_weight":
                bias = all_tensors[
                    f"_tag_transformer._decoder.layers.{layer_idx}.multihead_attn.in_proj_bias"
                ]
                (q_w, q_b), (k_w, k_b), (v_w, v_b) = _split_in_proj(
                    tensor, bias
                )
                decoder_sd[prefix + "cross_attn.q_proj.weight"] = q_w
                decoder_sd[prefix + "cross_attn.q_proj.bias"] = q_b
                decoder_sd[prefix + "cross_attn.k_proj.weight"] = k_w
                decoder_sd[prefix + "cross_attn.k_proj.bias"] = k_b
                decoder_sd[prefix + "cross_attn.v_proj.weight"] = v_w
                decoder_sd[prefix + "cross_attn.v_proj.bias"] = v_b
            elif param_path == "multihead_attn.in_proj_bias":
                pass  # handled above
            elif param_path.startswith("self_attn.out_proj."):
                suffix = param_path.replace("self_attn.out_proj.", "")
                decoder_sd[prefix + f"self_attn.out_proj.{suffix}"] = tensor
            elif param_path.startswith("multihead_attn.out_proj."):
                suffix = param_path.replace("multihead_attn.out_proj.", "")
                decoder_sd[prefix + f"cross_attn.out_proj.{suffix}"] = tensor
            else:
                # norm1/2/3, linear1/2 — pass through
                decoder_sd[prefix + param_path] = tensor

        # --- BBox decoder ---
        elif name.startswith("_bbox_decoder."):
            new_name = name.replace("_bbox_decoder.", "")
            bbox_sd[new_name] = tensor

    return encoder_sd, decoder_sd, bbox_sd
