"""Upload exported ONNX models to HuggingFace Hub.

Usage:
    python publish.py                     # publish all models
    python publish.py --model glm-ocr     # publish GLM-OCR only
    python publish.py --model tableformer # publish TableFormer only

Requires: pip install huggingface_hub
Login:    huggingface-cli login
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

SCRIPT_DIR = Path(__file__).parent
MODEL_CARDS_DIR = SCRIPT_DIR / "model_cards"

# HuggingFace repo IDs (must match Rust constants in models.rs)
GLM_OCR_REPO = "mgeorge412/glm_ocr"
TABLEFORMER_REPO = "mgeorge412/tableformer"


# GLM-OCR: source directory and files to upload (names match HF repo)
GLM_OCR_DIR = SCRIPT_DIR / "glm-ocr" / "model"
GLM_OCR_FILES = [
    "vision_encoder_mha.onnx",
    "vision_encoder_mha.onnx.data",
    "embedding.onnx",
    "embedding.onnx.data",
    "llm.onnx",
    "llm.onnx.data",
    "llm_decoder_gqa.onnx",
    "llm_decoder_gqa.onnx.data",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "config.json",
    "chat_template.jinja",
]

# TableFormer: source directory and file mapping (local name -> HF repo name)
TABLEFORMER_DIR = SCRIPT_DIR / "table_former" / "data"
TABLEFORMER_FILES = [
    "encoder.onnx",
    "decoder.onnx",
    "bbox_decoder.onnx",
]


def upload_model_card(api: HfApi, repo_id: str, readme_path: Path, license_path: Path) -> None:
    """Upload README.md and LICENSE to a HF repo."""
    if readme_path.exists():
        print(f"  Uploading README.md...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
    if license_path.exists():
        print(f"  Uploading LICENSE...")
        api.upload_file(
            path_or_fileobj=str(license_path),
            path_in_repo="LICENSE",
            repo_id=repo_id,
        )


def publish_glm_ocr(api: HfApi) -> None:
    api.create_repo(GLM_OCR_REPO, exist_ok=True)
    upload_model_card(
        api, GLM_OCR_REPO,
        MODEL_CARDS_DIR / "glm_ocr_README.md",
        MODEL_CARDS_DIR / "glm_ocr_LICENSE",
    )
    for filename in GLM_OCR_FILES:
        local_path = GLM_OCR_DIR / filename
        if not local_path.exists():
            print(f"  SKIP {filename} (not found at {local_path})")
            continue
        print(f"  Uploading {filename} ({local_path.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=GLM_OCR_REPO,
        )
    print(f"  Done: https://huggingface.co/{GLM_OCR_REPO}")


def publish_tableformer(api: HfApi) -> None:
    api.create_repo(TABLEFORMER_REPO, exist_ok=True)
    upload_model_card(
        api, TABLEFORMER_REPO,
        MODEL_CARDS_DIR / "tableformer_README.md",
        MODEL_CARDS_DIR / "tableformer_LICENSE",
    )
    for filename in TABLEFORMER_FILES:
        local_path = TABLEFORMER_DIR / filename
        if not local_path.exists():
            print(f"  SKIP {filename} (not found at {local_path})")
            continue
        print(f"  Uploading {filename} ({local_path.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=TABLEFORMER_REPO,
        )
    print(f"  Done: https://huggingface.co/{TABLEFORMER_REPO}")


PP_FORMULANET_REPO = "mgeorge412/pp_formulanet"
PP_FORMULANET_DIR = SCRIPT_DIR / "pp-formulanet" / "cuda" / "output"
PP_FORMULANET_FILES = [
    "encoder_fp16.onnx",
    "decoder_fp16_argmax.onnx",
    "unimernet_tokenizer.json",
]


def publish_pp_formulanet(api: HfApi) -> None:
    api.create_repo(PP_FORMULANET_REPO, exist_ok=True)
    upload_model_card(
        api, PP_FORMULANET_REPO,
        MODEL_CARDS_DIR / "pp_formulanet_README.md",
        MODEL_CARDS_DIR / "pp_formulanet_LICENSE",
    )
    for filename in PP_FORMULANET_FILES:
        local_path = PP_FORMULANET_DIR / filename
        if not local_path.exists():
            print(f"  SKIP {filename} (not found at {local_path})")
            continue
        print(f"  Uploading {filename} ({local_path.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=filename,
            repo_id=PP_FORMULANET_REPO,
        )
    print(f"  Done: https://huggingface.co/{PP_FORMULANET_REPO}")


PUBLISHERS = {
    "glm-ocr": ("GLM-OCR", publish_glm_ocr),
    "tableformer": ("TableFormer", publish_tableformer),
    "pp-formulanet": ("PP-FormulaNet", publish_pp_formulanet),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ONNX models to HuggingFace Hub")
    parser.add_argument(
        "--model",
        choices=list(PUBLISHERS.keys()),
        help="Publish only this model (default: all)",
    )
    args = parser.parse_args()

    api = HfApi()

    targets = [args.model] if args.model else list(PUBLISHERS.keys())
    for key in targets:
        label, publish_fn = PUBLISHERS[key]
        print(f"Publishing {label}...")
        publish_fn(api)


if __name__ == "__main__":
    main()
