"""Export ONNX models to the local cache directory for papers-extract.

This script dispatches to each model's own export script via subprocess
(each has its own uv environment with heavy deps like PyTorch).

Usage:
    python export.py                         # export all models
    python export.py --model glm-ocr         # export GLM-OCR only
    python export.py --model tableformer     # export TableFormer only
    python export.py --cache-dir PATH        # custom output directory

Default cache directory matches the Rust code:
    PAPERS_MODEL_DIR env var > platform cache dir/papers/models/
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def default_cache_dir() -> Path:
    """Match the Rust default_cache_dir() / layout_cache_dir() logic."""
    env = os.environ.get("PAPERS_MODEL_DIR")
    if env:
        return Path(env)
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "papers" / "models"


def export_glm_ocr(cache_dir: Path) -> None:
    """Export GLM-OCR models to cache_dir."""
    export_script = SCRIPT_DIR / "glm-ocr" / "cuda" / "export.py"
    if not export_script.exists():
        print(f"  ERROR: {export_script} not found", file=sys.stderr)
        sys.exit(1)

    # GLM-OCR export.py supports --output-dir
    cmd = [
        sys.executable, str(export_script),
        "--bf16",
        "--output-dir", str(cache_dir),
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(export_script.parent))


def export_tableformer(cache_dir: Path) -> None:
    """Export TableFormer models to cache_dir.

    The export script writes to a local data/ directory with bare names
    (encoder.onnx, decoder.onnx, bbox_decoder.onnx). We run it then
    copy outputs to the cache dir.
    """
    export_script = SCRIPT_DIR / "table_former" / "export.py"
    if not export_script.exists():
        print(f"  ERROR: {export_script} not found", file=sys.stderr)
        sys.exit(1)

    # Run the export (outputs to table_former/data/)
    cmd = [sys.executable, str(export_script)]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(export_script.parent))

    # Copy outputs to cache dir (bare names, no prefix)
    source_dir = export_script.parent / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ["encoder.onnx", "decoder.onnx", "bbox_decoder.onnx"]:
        src = source_dir / name
        dst = cache_dir / name
        if src.exists():
            print(f"  Copying {name} -> {dst}")
            shutil.copy2(src, dst)
        else:
            print(f"  WARNING: {src} not found", file=sys.stderr)


EXPORTERS = {
    "glm-ocr": ("GLM-OCR", export_glm_ocr),
    "tableformer": ("TableFormer", export_tableformer),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ONNX models to the local cache directory"
    )
    parser.add_argument(
        "--model",
        choices=list(EXPORTERS.keys()),
        help="Export only this model (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {default_cache_dir()})",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    targets = [args.model] if args.model else list(EXPORTERS.keys())
    for key in targets:
        label, export_fn = EXPORTERS[key]
        print(f"\nExporting {label}...")
        export_fn(cache_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
