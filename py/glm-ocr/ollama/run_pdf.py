"""Run GLM-OCR via Ollama on every page of a PDF (parallel requests).

Sends all page requests concurrently. Ollama queues them on the GPU.

Usage:
    uv run python run_pdf.py ../../../data/vbd.pdf
    uv run python run_pdf.py --dpi 150 --pages 1-3 ../../../data/vbd.pdf
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

PROMPT = (
    "Recognize the text in the image and output in Markdown format. "
    "Preserve the original layout (headings/paragraphs/tables/formulas). "
    "Do not fabricate content that does not exist in the image."
)


def pdf_to_images(pdf_path, dpi=150):
    """Convert PDF pages to PNG bytes using PyMuPDF."""
    import fitz

    doc = fitz.open(str(pdf_path))
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pages.append(pix.tobytes("png"))
    doc.close()
    return pages


def ocr_page(idx, img_bytes, num_ctx=16384):
    """Send a page image to Ollama GLM-OCR, return (idx, text, stats)."""
    img_b64 = base64.b64encode(img_bytes).decode()

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
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    elapsed = time.perf_counter() - t0

    text = data.get("message", {}).get("content", "")
    stats = {
        "eval_tokens": data.get("eval_count", 0),
        "eval_ns": data.get("eval_duration", 0),
        "elapsed": elapsed,
    }
    return idx, text, stats


def main():
    parser = argparse.ArgumentParser(description="Run GLM-OCR on a PDF via Ollama")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--pages", type=str, default=None,
                        help="Page range, e.g. '1-5' or '3' (1-indexed)")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=16,
                        help="Max parallel requests (default: 16)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Rendering PDF at {args.dpi} DPI...", file=sys.stderr)
    all_pages = pdf_to_images(pdf_path, dpi=args.dpi)
    print(f"  {len(all_pages)} pages", file=sys.stderr)

    if args.pages:
        if '-' in args.pages:
            start, end = args.pages.split('-', 1)
            start, end = int(start) - 1, int(end)
            indices = list(range(start, min(end, len(all_pages))))
        else:
            indices = [int(args.pages) - 1]
    else:
        indices = list(range(len(all_pages)))

    print(f"  Sending {len(indices)} pages in parallel...", file=sys.stderr)

    t0_total = time.perf_counter()
    results = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(ocr_page, idx, all_pages[idx], args.num_ctx): idx
            for idx in indices
        }
        for future in as_completed(futures):
            idx, text, stats = future.result()
            results[idx] = text
            et = stats["eval_tokens"]
            ed = stats["eval_ns"]
            tps = et / (ed / 1e9) if ed else 0
            print(f"  Page {idx+1}: {et} tokens, {stats['elapsed']:.1f}s ({tps:.0f} tok/s)",
                  file=sys.stderr)

    total = time.perf_counter() - t0_total

    # Output in page order
    for idx in indices:
        print(f"<!-- Page {idx+1} -->")
        print(results[idx])
        print()

    print(f"\nDone: {len(indices)} pages in {total:.1f}s "
          f"({total/len(indices):.1f}s/page)", file=sys.stderr)


if __name__ == "__main__":
    main()
