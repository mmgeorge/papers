"""Extract PP-FormulaNet Plus-L weights from PaddlePaddle training checkpoint.

Downloads the official training checkpoint, loads with paddle.load(),
filters encoder and decoder parameters, transposes Linear weights
(Paddle [in,out] -> PyTorch [out,in]), and saves as .npz files.
"""

import os
import sys
import urllib.request

import numpy as np

# PP-FormulaNet Plus-L training checkpoint
CHECKPOINT_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams"
)
CHECKPOINT_FILENAME = "PP-FormulaNet_plus-L_pretrained.pdparams"

# --- Decoder weight config ---

# Linear weight suffixes that need transposition: Paddle [in,out] -> PyTorch [out,in]
# LayerNorm, Embedding, and biases are copied as-is.
DECODER_TRANSPOSE_SUFFIXES = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "out_proj.weight",
    "fc1.weight",
    "fc2.weight",
    "enc_to_dec_proj.weight",
    "lm_head.weight",
]

# Prefixes that identify decoder parameters (vs encoder/backbone).
DECODER_PREFIX_MAP = {
    "head.enc_to_dec_proj.": "enc_to_dec_proj.",
    "head.decoder.model.decoder.": "model.decoder.",
    "head.decoder.lm_head.": "lm_head.",
}

# --- Encoder weight config ---

# Encoder Linear weights that need transposition.
ENCODER_TRANSPOSE_SUFFIXES = [
    "attn.qkv.weight",
    "attn.proj.weight",
    "mlp.lin1.weight",
    "mlp.lin2.weight",
    "mm_projector_vary.weight",
]

# Prefixes that identify encoder parameters.
ENCODER_PREFIX_MAP = {
    "backbone.vision_tower_high.": "vision_tower_high.",
    "backbone.mm_projector_vary.": "mm_projector_vary.",
}


def download_checkpoint(output_dir):
    """Download the training checkpoint if not already present.

    Args:
        output_dir: Directory to save the checkpoint file.

    Returns:
        Path to the downloaded checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CHECKPOINT_FILENAME)

    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"Checkpoint already exists: {path} ({size_mb:.1f} MB)")
        return path

    print(f"Downloading checkpoint...")
    print(f"  URL: {CHECKPOINT_URL}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    urllib.request.urlretrieve(CHECKPOINT_URL, path, progress_hook)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n  Downloaded: {size_mb:.1f} MB")
    return path


def load_paddle_weights(path):
    """Load PaddlePaddle checkpoint and convert to numpy arrays.

    Falls back to pickle if PaddlePaddle is not installed (.pdparams
    files are just pickled dicts of numpy arrays).
    """
    try:
        import paddle
        print(f"\nLoading checkpoint with paddle.load()...")
        state_dict = paddle.load(path)
    except ImportError:
        import pickle
        print(f"\nLoading checkpoint with pickle (paddle not installed)...")
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

    # Convert to numpy arrays
    weights = {}
    for k, v in state_dict.items():
        if isinstance(v, np.ndarray):
            weights[k] = v
        elif hasattr(v, "numpy"):
            weights[k] = v.numpy()
        else:
            print(f"  WARNING: skipping {k} (type={type(v).__name__})")
    return weights


def extract_encoder_weights(weights):
    """Filter to encoder-only params and transpose Linear weights."""
    encoder_weights = {}

    for name, arr in weights.items():
        clean_name = None
        for paddle_prefix, pytorch_prefix in ENCODER_PREFIX_MAP.items():
            if name.startswith(paddle_prefix):
                clean_name = pytorch_prefix + name[len(paddle_prefix):]
                break

        if clean_name is None:
            continue

        # Transpose Linear weights (not Conv2D, LayerNorm, pos embeddings, or biases)
        needs_transpose = any(clean_name.endswith(s) for s in ENCODER_TRANSPOSE_SUFFIXES)
        if needs_transpose and arr.ndim == 2:
            arr = arr.T

        encoder_weights[clean_name] = arr

    return encoder_weights


def extract_decoder_weights(weights):
    """Filter to decoder-only params and transpose Linear weights."""
    decoder_weights = {}

    for name, arr in weights.items():
        clean_name = None
        for paddle_prefix, pytorch_prefix in DECODER_PREFIX_MAP.items():
            if name.startswith(paddle_prefix):
                clean_name = pytorch_prefix + name[len(paddle_prefix):]
                break

        if clean_name is None:
            continue

        # Transpose Linear weights
        needs_transpose = any(clean_name.endswith(s) for s in DECODER_TRANSPOSE_SUFFIXES)
        if needs_transpose and arr.ndim == 2:
            arr = arr.T

        decoder_weights[clean_name] = arr

    return decoder_weights


def convert_and_save(output_dir):
    """Full pipeline: download checkpoint -> extract -> save .npz files.

    Args:
        output_dir: Directory for checkpoint and output .npz files.

    Returns:
        (encoder_path, decoder_path) tuple of saved .npz file paths.
    """
    encoder_path = os.path.join(output_dir, "encoder_weights.npz")
    decoder_path = os.path.join(output_dir, "decoder_weights.npz")

    checkpoint_path = download_checkpoint(output_dir)
    weights = load_paddle_weights(checkpoint_path)

    # Extract and save encoder weights
    encoder_weights = extract_encoder_weights(weights)
    if not encoder_weights:
        print("\nERROR: No encoder parameters found!")
        sys.exit(1)
    np.savez(encoder_path, **encoder_weights)
    size_mb = os.path.getsize(encoder_path) / 1e6
    print(f"Saved: {encoder_path} ({size_mb:.1f} MB, {len(encoder_weights)} params)")

    # Extract and save decoder weights
    decoder_weights = extract_decoder_weights(weights)
    if not decoder_weights:
        print("\nERROR: No decoder parameters found!")
        sys.exit(1)
    np.savez(decoder_path, **decoder_weights)
    size_mb = os.path.getsize(decoder_path) / 1e6
    print(f"Saved: {decoder_path} ({size_mb:.1f} MB, {len(decoder_weights)} params)")

    return encoder_path, decoder_path
