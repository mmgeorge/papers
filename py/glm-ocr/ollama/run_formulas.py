"""Run GLM-OCR via Ollama on formula images (parallel requests).

Usage:
    uv run python run_formulas.py [formulas_dir]
    uv run python run_formulas.py ../../../../test-extract/formulas
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OLLAMA_URL = "http://localhost:11434/api/chat"
PROMPT = "Formula Recognition:"


def ocr_formula(img_path, num_ctx=4096):
    """Send a formula image to Ollama GLM-OCR, return (name, text, stats)."""
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = json.dumps({
        "model": "glm-ocr",
        "messages": [{
            "role": "user",
            "content": PROMPT,
            "images": [img_b64],
        }],
        "stream": False,
        "options": {"num_ctx": num_ctx},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0

    text = data.get("message", {}).get("content", "")
    stats = {
        "eval_tokens": data.get("eval_count", 0),
        "eval_ns": data.get("eval_duration", 0),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "elapsed": elapsed,
    }
    return img_path.name, text, stats


def main():
    parser = argparse.ArgumentParser(description="Run GLM-OCR on formula images via Ollama")
    parser.add_argument("formulas_dir", nargs="?", default="../../../../test-extract/formulas")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    formulas_dir = Path(args.formulas_dir)
    image_paths = sorted(formulas_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG files found in {formulas_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Sending {len(image_paths)} formulas in parallel...", file=sys.stderr)

    t0_total = time.perf_counter()
    results = {}
    total_eval_tokens = 0
    total_eval_ns = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(ocr_formula, p, args.num_ctx): p
            for p in image_paths
        }
        done = 0
        for future in as_completed(futures):
            name, text, stats = future.result()
            # Strip $$ wrapping
            latex = text.strip()
            if latex.startswith("$$") and latex.endswith("$$"):
                latex = latex[2:-2].strip()
            elif latex.startswith("$") and latex.endswith("$"):
                latex = latex[1:-1].strip()
            results[name] = latex
            total_eval_tokens += stats["eval_tokens"]
            total_eval_ns += stats["eval_ns"]
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(image_paths)} done...", file=sys.stderr)

    total = time.perf_counter() - t0_total
    tps = total_eval_tokens / (total_eval_ns / 1e9) if total_eval_ns else 0

    # Output in sorted order
    for name in sorted(results.keys()):
        print(f"{name}\t{results[name]}")

    print(f"\nDone: {len(image_paths)} formulas in {total:.1f}s "
          f"({total/len(image_paths):.2f}s/formula, "
          f"{total_eval_tokens} tokens, {tps:.0f} tok/s)", file=sys.stderr)


if __name__ == "__main__":
    main()
