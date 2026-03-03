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


# Normalization constants from UniMERNet training pipeline.
# The model was trained on formula images (mostly white background, black text),
# so the dataset-specific mean is high and std is low.
# Source: UniMERNetTestTransform in ppocr/data/imaug/unimernet_aug.py and
#         opendatalab/UniMERNet unimernet/processors/formula_processor.py
# Both use: albumentations.Normalize((0.7931, 0.7931, 0.7931), (0.1738, 0.1738, 0.1738))
NORM_MEAN = 0.7931
NORM_STD = 0.1738

# Content crop threshold (after min-max normalization to 0-255).
# PaddleOCR/UniMERNet use 200, treating anything darker than ~78% white as content.
# More permissive than 127, capturing anti-aliased edges and thin strokes.
# Source: crop_margin() in PaddleOCR unimernet_aug.py and UniMERNet formula_processor.py
CROP_THRESHOLD = 200


def preprocess_image(img_path, dtype=np.float32):
    """Preprocess a formula image for PP-FormulaNet encoder.

    Pipeline matches PaddleOCR's UniMERNetImgDecode + UniMERNetTestTransform +
    LatexImageFormat, which is what the model was trained on.

    Steps: RGB load -> BT.601 grayscale -> content crop (threshold 200) ->
    bilinear resize to fit TARGET_SIZE -> center-pad to TARGET_SIZExTARGET_SIZE
    -> BT.601 luminance -> normalize (mean=0.7931, std=0.1738).

    Args:
        img_path: Path to the input image.
        dtype: Output dtype (np.float32 for DirectML, np.float16 for CUDA).

    Returns:
        Preprocessed image as [1, 1, TARGET_SIZE, TARGET_SIZE] numpy array.
    """
    img = Image.open(img_path).convert("RGB")

    # Content crop using BT.601 luminance (PIL "L" mode), matching PaddleOCR crop_margin()
    gray = np.array(img.convert("L")).astype(np.float32)
    min_val, max_val = gray.min(), gray.max()
    if max_val > min_val:
        gray_norm = (gray - min_val) / (max_val - min_val) * 255.0
    else:
        gray_norm = gray
    binary = (gray_norm < CROP_THRESHOLD).astype(np.uint8)
    coords = np.argwhere(binary)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img.crop((x0, y0, x1, y1))

    # Resize with BILINEAR interpolation, matching UniMERNetImgDecode.resize()
    # which uses img.resize(..., resample=2) i.e. PIL.Image.BILINEAR
    w, h = img.size
    scale = TARGET_SIZE / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    padded = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    padded.paste(img, ((TARGET_SIZE - img.width) // 2, (TARGET_SIZE - img.height) // 2))

    # BT.601 luminance conversion + dataset-specific normalization
    arr = np.array(padded).astype(np.float32)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    normalized = (lum / 255.0 - NORM_MEAN) / NORM_STD
    return normalized[np.newaxis, np.newaxis, :, :].astype(dtype)


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
