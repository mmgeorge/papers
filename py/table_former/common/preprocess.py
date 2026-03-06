"""Image preprocessing matching docling's TableFormer pipeline."""

import numpy as np
from PIL import Image

# From tm_config.json (dataset-specific, NOT ImageNet defaults)
MEAN = np.array([0.94247851, 0.94254675, 0.94292611], dtype=np.float32)
STD = np.array([0.17910956, 0.17940403, 0.17931663], dtype=np.float32)
IMAGE_SIZE = 448


def preprocess(image_path: str) -> np.ndarray:
    """Preprocess image for TableFormer inference.

    Matches docling tf_predictor._prepare_image():
    - BGR channel order (OpenCV convention)
    - Normalize: (pixel/255 - mean) / std
    - transpose(2, 1, 0): HWC → CWH (spatial transpose, docling convention)

    Returns: [1, 3, 448, 448] float32
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] RGB float 0-1

    # RGB → BGR (match docling's OpenCV convention)
    arr = arr[:, :, ::-1].copy()

    # Normalize per-channel
    arr = (arr - MEAN) / STD

    # HWC → CWH (docling's transpose(2, 1, 0), NOT standard CHW)
    arr = arr.transpose(2, 1, 0)

    return arr[np.newaxis].astype(np.float32)  # [1, 3, 448, 448]
