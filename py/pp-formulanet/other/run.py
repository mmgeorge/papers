"""Run PP-FormulaNet inference with CPU (session.run, FP32).

Uses basic session.run() with CPU-side argmax. No IOBinding or CUDA graphs.

Usage:
    python run.py image.png [image2.png ...]
    python run.py --formulas-dir path/to/formulas/
"""

import argparse
import os
import sys
import time

os.environ["ORT_LOG_LEVEL"] = "3"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import onnxruntime as ort
from pathlib import Path
from tokenizers import Tokenizer

from common.preprocess import preprocess_image, decode_tokens, BOS_ID, EOS_ID, N_LAYERS, N_HEADS, HEAD_DIM, MAX_SEQ

ENCODER_PATH = "output/encoder.onnx"
DECODER_PATH = "output/decoder.onnx"
TOKENIZER_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "papers", "models", "unimernet_tokenizer.json"
)


def run_encoder(enc_sess, image):
    results = enc_sess.run(None, {enc_sess.get_inputs()[0].name: image})
    for r in results:
        if len(r.shape) == 3 and r.shape[-1] == 1024:
            return r
    return results[0]


def decode_basic(dec_sess, encoder_hidden, max_steps=MAX_SEQ):
    """Basic session.run with fixed-size KV buffers."""
    batch = encoder_hidden.shape[0]

    past_kv = {}
    for layer in range(N_LAYERS):
        for kv in ["key", "value"]:
            past_kv[f"past_key_values.{layer}.{kv}"] = \
                np.zeros((batch, N_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float32)

    input_ids = np.array([[BOS_ID]], dtype=np.int64)
    tokens = [BOS_ID]

    for s in range(max_steps):
        feed = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden,
            "step": np.array([s], dtype=np.int64),
            **past_kv,
        }
        outputs = dec_sess.run(None, feed)

        next_token = int(np.argmax(outputs[0][0, -1, :]))
        tokens.append(next_token)
        if next_token == EOS_ID:
            break

        out_idx = 1
        for layer in range(N_LAYERS):
            for kv in ["key", "value"]:
                past_kv[f"past_key_values.{layer}.{kv}"] = outputs[out_idx]
                out_idx += 1
        input_ids = np.array([[next_token]], dtype=np.int64)

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Run PP-FormulaNet with CPU")
    parser.add_argument("images", nargs="*", help="Image files to process")
    parser.add_argument("--formulas-dir", type=str, help="Directory of formula images")
    parser.add_argument("--encoder", default=ENCODER_PATH, help="Encoder ONNX path")
    parser.add_argument("--decoder", default=DECODER_PATH, help="Decoder ONNX path")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH, help="Tokenizer path")
    args = parser.parse_args()

    # Collect image paths
    image_paths = []
    if args.formulas_dir:
        image_paths = sorted(Path(args.formulas_dir).glob("*.png"))
    if args.images:
        image_paths.extend(Path(p) for p in args.images)
    if not image_paths:
        parser.error("No images specified. Use positional args or --formulas-dir.")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Create sessions
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    print("Loading encoder...")
    enc_sess = ort.InferenceSession(args.encoder, sess_options=opts, providers=providers)
    print("Loading decoder...")
    dec_sess = ort.InferenceSession(args.decoder, sess_options=opts, providers=providers)
    print(f"Active: {dec_sess.get_providers()}")

    # Warmup
    print("Warmup...")
    warmup_img = preprocess_image(str(image_paths[0]), dtype=np.float32)
    warmup_enc = run_encoder(enc_sess, warmup_img)
    _ = decode_basic(dec_sess, warmup_enc)

    # Run all formulas
    n = len(image_paths)
    print(f"\nRunning on {n} formulas (CPU + FP32)\n")
    print(f"{'Image':<12} {'Tokens':>6} {'Encode':>10} {'Decode':>10} {'Total':>10}  {'LaTeX'}")
    print("-" * 120)

    total_encode = 0
    total_decode = 0
    total_tokens = 0

    for img_path in image_paths:
        image = preprocess_image(str(img_path), dtype=np.float32)

        t0 = time.perf_counter()
        enc_hidden = run_encoder(enc_sess, image)
        t_enc = time.perf_counter() - t0

        t1 = time.perf_counter()
        tokens = decode_basic(dec_sess, enc_hidden)
        t_dec = time.perf_counter() - t1

        latex = decode_tokens(tokenizer, tokens)
        n_tokens = len(tokens)

        total_encode += t_enc
        total_decode += t_dec
        total_tokens += n_tokens

        print(f"{img_path.stem:<12} {n_tokens:>6} {t_enc*1000:>8.0f}ms {t_dec*1000:>8.0f}ms {(t_enc+t_dec)*1000:>8.0f}ms  {latex[:80]}")

    total_time = total_encode + total_decode
    print("-" * 120)
    print(f"{'TOTAL':<12} {total_tokens:>6} {total_encode*1000:>8.0f}ms {total_decode*1000:>8.0f}ms {total_time*1000:>8.0f}ms")
    print(f"{'AVERAGE':<12} {total_tokens//n:>6} {total_encode/n*1000:>8.0f}ms {total_decode/n*1000:>8.0f}ms {total_time/n*1000:>8.0f}ms")
    print(f"\n{n} formulas, {total_tokens} total tokens, {total_time*1000:.0f}ms total ({total_time/n*1000:.0f}ms avg)")


if __name__ == "__main__":
    main()
