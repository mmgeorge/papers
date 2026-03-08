"""Run PP-FormulaNet inference with CUDA graphs and IOBinding.

Uses native FP16 I/O, GPU-side ArgMax, and CUDA graphs for optimal
performance. Only 8 bytes (one int64 token ID) cross GPU->CPU per step.

Usage:
    python run.py image.png [image2.png ...]
    python run.py --formulas-dir path/to/formulas/
"""

import argparse
import ctypes
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

ENCODER_PATH = "output/encoder_fp16.onnx"
DECODER_PATH = "output/decoder_fp16_argmax.onnx"
TOKENIZER_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "papers", "models", "unimernet_tokenizer.json"
)

MEMCPY_H2D = 1
MEMCPY_D2H = 2


def find_cudart():
    """Find cudart DLL dynamically."""
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        bin_dir = os.path.join(cuda_path, "bin")
        for f in os.listdir(bin_dir):
            if f.startswith("cudart64_") and f.endswith(".dll"):
                return os.path.join(bin_dir, f)
    # Fallback: common install paths
    for ver in ["12.8", "12.6", "12.4", "12.2", "12.0"]:
        path = rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{ver}\bin\cudart64_12.dll"
        if os.path.exists(path):
            return path
    raise RuntimeError("Could not find cudart64 DLL. Set CUDA_PATH environment variable.")


def main():
    parser = argparse.ArgumentParser(description="Run PP-FormulaNet with CUDA graphs")
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

    # Load cudart
    cudart_path = find_cudart()
    cudart = ctypes.CDLL(cudart_path)
    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int
    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    def gpu_update(ort_value, np_array):
        cudart.cudaMemcpy(
            ctypes.c_void_p(ort_value.data_ptr()),
            ctypes.c_void_p(np_array.ctypes.data),
            np_array.nbytes, MEMCPY_H2D)

    def gpu_read_int64(ort_value, cpu_buf):
        cudart.cudaMemcpy(
            ctypes.c_void_p(cpu_buf.ctypes.data),
            ctypes.c_void_p(ort_value.data_ptr()),
            cpu_buf.nbytes, MEMCPY_D2H)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Create sessions
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    enc_sess = ort.InferenceSession(args.encoder, sess_options=opts,
                                     providers=["CUDAExecutionProvider"])
    dec_sess = ort.InferenceSession(
        args.decoder, sess_options=opts,
        providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})])

    batch = 1
    # Pre-allocate persistent GPU buffers for decoder
    ov_input_ids = ort.OrtValue.ortvalue_from_numpy(
        np.array([[BOS_ID]], dtype=np.int64), "cuda", 0)
    ov_step = ort.OrtValue.ortvalue_from_numpy(
        np.array([0], dtype=np.int64), "cuda", 0)

    kv_names, present_names = [], []
    kv_ovs = {}
    for layer in range(N_LAYERS):
        for kv in ["key", "value"]:
            past_name = f"past_key_values.{layer}.{kv}"
            present_name = f"present_key_values.{layer}.{kv}"
            kv_names.append(past_name)
            present_names.append(present_name)
            kv_ovs[past_name] = ort.OrtValue.ortvalue_from_numpy(
                np.zeros((batch, N_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float16), "cuda", 0)

    ov_logits = ort.OrtValue.ortvalue_from_shape_and_type(
        [batch, 1, 50000], np.float16, "cuda", 0)
    ov_next_token = ort.OrtValue.ortvalue_from_shape_and_type(
        [1], np.int64, "cuda", 0)

    cpu_input_ids = np.array([[BOS_ID]], dtype=np.int64)
    cpu_step = np.array([0], dtype=np.int64)
    cpu_token_buf = np.array([0], dtype=np.int64)

    def decode_formula(enc_hidden_fp16):
        ov_enc = ort.OrtValue.ortvalue_from_numpy(enc_hidden_fp16, "cuda", 0)
        for name in kv_names:
            kv_ovs[name] = ort.OrtValue.ortvalue_from_numpy(
                np.zeros((batch, N_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float16), "cuda", 0)
        tokens = [BOS_ID]
        for s in range(MAX_SEQ):
            cpu_input_ids[0, 0] = tokens[-1]
            cpu_step[0] = s
            gpu_update(ov_input_ids, cpu_input_ids)
            gpu_update(ov_step, cpu_step)
            io = dec_sess.io_binding()
            io.bind_ortvalue_input("input_ids", ov_input_ids)
            io.bind_ortvalue_input("step", ov_step)
            io.bind_ortvalue_input("encoder_hidden_states", ov_enc)
            for name in kv_names:
                io.bind_ortvalue_input(name, kv_ovs[name])
            io.bind_ortvalue_output("logits", ov_logits)
            for i, pn in enumerate(present_names):
                io.bind_ortvalue_output(pn, kv_ovs[kv_names[i]])
            io.bind_ortvalue_output("next_token", ov_next_token)
            dec_sess.run_with_iobinding(io)
            gpu_read_int64(ov_next_token, cpu_token_buf)
            next_token = int(cpu_token_buf[0])
            tokens.append(next_token)
            if next_token == EOS_ID:
                break
        return tokens

    # Warmup (3 full formulas)
    print("Warmup...")
    for _ in range(3):
        img = preprocess_image(str(image_paths[0]), dtype=np.float16)
        enc_out = enc_sess.run(None, {enc_sess.get_inputs()[0].name: img})
        _ = decode_formula(enc_out[0])
    cudart.cudaDeviceSynchronize()

    # Run all formulas
    print(f"\nRunning on {len(image_paths)} formulas (CUDA graphs + FP16)\n")
    print(f"{'Image':<12} {'Tokens':>6} {'Encode':>10} {'Decode':>10} {'Total':>10}  {'LaTeX'}")
    print("-" * 120)

    total_encode = 0
    total_decode = 0
    total_tokens = 0

    for img_path in image_paths:
        image = preprocess_image(str(img_path), dtype=np.float16)

        t0 = time.perf_counter()
        enc_out = enc_sess.run(None, {enc_sess.get_inputs()[0].name: image})
        cudart.cudaDeviceSynchronize()
        t_enc = time.perf_counter() - t0

        t1 = time.perf_counter()
        tokens = decode_formula(enc_out[0])
        cudart.cudaDeviceSynchronize()
        t_dec = time.perf_counter() - t1

        latex = decode_tokens(tokenizer, tokens)
        n_tokens = len(tokens)

        total_encode += t_enc
        total_decode += t_dec
        total_tokens += n_tokens

        print(f"{img_path.stem:<12} {n_tokens:>6} {t_enc*1000:>8.0f}ms {t_dec*1000:>8.0f}ms {(t_enc+t_dec)*1000:>8.0f}ms  {latex[:80]}")

    total_time = total_encode + total_decode
    n = len(image_paths)
    print("-" * 120)
    print(f"{'TOTAL':<12} {total_tokens:>6} {total_encode*1000:>8.0f}ms {total_decode*1000:>8.0f}ms {total_time*1000:>8.0f}ms")
    print(f"{'AVERAGE':<12} {total_tokens//n:>6} {total_encode/n*1000:>8.0f}ms {total_decode/n*1000:>8.0f}ms {total_time/n*1000:>8.0f}ms")
    print(f"\n{n} formulas, {total_tokens} total tokens, {total_time*1000:.0f}ms total ({total_time/n*1000:.0f}ms avg)")


if __name__ == "__main__":
    main()
