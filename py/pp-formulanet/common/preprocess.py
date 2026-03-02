"""Shared preprocessing and decoding helpers for PP-FormulaNet inference."""

import numpy as np
from PIL import Image

# Model constants
TARGET_SIZE = 768
BOS_ID = 0
EOS_ID = 2
N_LAYERS = 8
N_HEADS = 16
HEAD_DIM = 32
MAX_SEQ = 512


def preprocess_image(img_path, dtype=np.float32):
    """Preprocess a formula image for PP-FormulaNet encoder.

    Pipeline: RGB load -> grayscale threshold -> content crop -> scale to
    fit TARGET_SIZE -> center-pad to TARGET_SIZExTARGET_SIZE -> normalize.

    Args:
        img_path: Path to the input image.
        dtype: Output dtype (np.float32 for DirectML, np.float16 for CUDA).

    Returns:
        Preprocessed image as [1, 1, TARGET_SIZE, TARGET_SIZE] numpy array.
    """
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    gray = np.mean(img_array, axis=2)
    min_val, max_val = gray.min(), gray.max()
    if max_val > min_val:
        gray_norm = (gray - min_val) / (max_val - min_val) * 255.0
    else:
        gray_norm = gray
    binary = (gray_norm < 127).astype(np.uint8)
    coords = np.argwhere(binary)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img.crop((x0, y0, x1, y1))
    w, h = img.size
    scale = TARGET_SIZE / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    padded.paste(img, ((TARGET_SIZE - img.width) // 2, (TARGET_SIZE - img.height) // 2))
    arr = np.array(padded).astype(np.float32)
    gray_f = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gray_f = (gray_f / 255.0 - 0.5) / 0.5
    return gray_f[np.newaxis, np.newaxis, :, :].astype(dtype)


def decode_tokens(tokenizer, token_ids):
    """Convert token IDs to LaTeX string.

    Args:
        tokenizer: A tokenizers.Tokenizer instance.
        token_ids: List of integer token IDs.

    Returns:
        Decoded LaTeX string.
    """
    vocab_size = tokenizer.get_vocab_size()
    result = []
    for tid in token_ids:
        if tid == EOS_ID:
            break
        if tid == BOS_ID or tid < 0 or tid >= vocab_size:
            continue
        result.append(int(tid))
    return tokenizer.decode(result, skip_special_tokens=True) if result else ""
