"""Run GLM-OCR inference with CPU (session.run, FP32).

Uses basic session.run() with CPU-side argmax. No IOBinding or CUDA graphs.
Uses llm.onnx for both prefill and decode steps with growing KV cache.

Usage:
  uv run python run.py --image path/to/image.png
  uv run python run.py --image path/to/image.png --prompt "Text Recognition:"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoProcessor, AutoConfig

from common.helpers import (
    NUM_LAYERS,
    NUM_KV_HEADS,
    HEAD_DIM,
    compute_vision_pos_ids,
    build_position_ids,
    detect_dtype,
)

DEFAULT_MODEL_DIR = "../model"
DEFAULT_PROMPT = "Formula Recognition:"
DEFAULT_MAX_TOKENS = 512


def generate(sessions, processor, config, image, prompt_text, max_new_tokens):
    """Run full generation: preprocess -> vision -> embed -> prefill -> decode."""
    image_token_id = config.image_token_id
    eos_token_ids = config.text_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    # 1. Preprocess
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].numpy()
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    image_grid_thw = inputs["image_grid_thw"].numpy().astype(np.int64)

    batch_size, seq_len = input_ids.shape

    # 2. Build M-RoPE position IDs
    position_ids = build_position_ids(input_ids, image_grid_thw, image_token_id)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    # 3. Vision encoder
    t0 = time.perf_counter()
    vis_pos_ids, max_grid_size = compute_vision_pos_ids(image_grid_thw)
    vis_dtype = detect_dtype(sessions["vision_encoder"])
    vision_output = sessions["vision_encoder"].run(None, {
        "pixel_values": pixel_values.astype(vis_dtype),
        "pos_ids": vis_pos_ids.astype(np.int64),
        "max_grid_size": np.array(max_grid_size, dtype=np.int64),
    })[0]
    t_vision = time.perf_counter() - t0

    # 4. Token embedding
    token_embeds = sessions["embedding"].run(
        None, {"input_ids": input_ids.astype(np.int64)}
    )[0]

    # 5. Merge vision embeddings into token embeddings
    inputs_embeds = token_embeds.copy()
    image_mask = (input_ids == image_token_id)
    vision_flat = vision_output.reshape(-1, vision_output.shape[-1])
    inputs_embeds[image_mask] = vision_flat.astype(inputs_embeds.dtype)

    float_dtype = detect_dtype(sessions["llm"])

    # 6. Prefill
    t0 = time.perf_counter()
    prefill_inputs = {
        "inputs_embeds": inputs_embeds.astype(float_dtype),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    for layer in range(NUM_LAYERS):
        prefill_inputs[f"past_key_{layer}"] = np.zeros(
            (batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=float_dtype
        )
        prefill_inputs[f"past_value_{layer}"] = np.zeros(
            (batch_size, NUM_KV_HEADS, 0, HEAD_DIM), dtype=float_dtype
        )

    outputs = sessions["llm"].run(None, prefill_inputs)
    logits = outputs[0]
    kv_cache = outputs[1:]

    t_prefill = time.perf_counter() - t0

    # 7. Decode loop
    t0 = time.perf_counter()
    generated_tokens = []
    current_pos = position_ids[:, :, -1:] + 1

    for step in range(max_new_tokens):
        next_token = int(np.argmax(logits[:, -1, :].astype(np.float32), axis=-1)[0])
        generated_tokens.append(next_token)

        if next_token in eos_token_ids:
            break

        next_ids = np.array([[next_token]], dtype=np.int64)
        next_embeds = sessions["embedding"].run(None, {"input_ids": next_ids})[0]

        next_pos = current_pos + step
        total_len = seq_len + step + 1
        decode_mask = np.ones((batch_size, total_len), dtype=np.int64)

        decode_inputs = {
            "inputs_embeds": next_embeds.astype(float_dtype),
            "attention_mask": decode_mask,
            "position_ids": next_pos,
        }
        for layer in range(NUM_LAYERS):
            decode_inputs[f"past_key_{layer}"] = kv_cache[layer * 2].astype(float_dtype)
            decode_inputs[f"past_value_{layer}"] = kv_cache[layer * 2 + 1].astype(float_dtype)

        outputs = sessions["llm"].run(None, decode_inputs)
        logits = outputs[0]
        kv_cache = outputs[1:]

        # Repetition detection
        if len(generated_tokens) >= 10:
            last_10 = generated_tokens[-10:]
            if all(t == last_10[-1] for t in last_10):
                break

    t_decode = time.perf_counter() - t0
    num_tokens = len(generated_tokens)
    tok_per_sec = (num_tokens - 1) / t_decode if t_decode > 0 and num_tokens > 1 else 0

    print(f"  Vision: {t_vision:.3f}s | Prefill: {t_prefill:.3f}s | "
          f"Decode: {t_decode:.2f}s ({tok_per_sec:.1f} tok/s) | "
          f"Tokens: {num_tokens}")

    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="GLM-OCR CPU/CoreML inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    llm_path = model_dir / "llm.onnx"
    if not llm_path.exists():
        print(f"No LLM model found in {model_dir}")
        print("Run export.py first.")
        sys.exit(1)

    required = ["vision_encoder.onnx", "embedding.onnx"]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        print(f"Missing: {', '.join(missing)} in {model_dir}")
        sys.exit(1)

    print(f"Loading processor from {model_dir} ...")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    config = AutoConfig.from_pretrained(str(model_dir))

    # Session options
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = ["CPUExecutionProvider"]

    # Vision encoder needs ORT_ENABLE_EXTENDED (mixed-type FP16 graph)
    vis_opts = ort.SessionOptions()
    vis_opts.log_severity_level = 3
    vis_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    print(f"Loading ONNX sessions (LLM: {llm_path.name}) ...")
    sessions = {
        "vision_encoder": ort.InferenceSession(
            str(model_dir / "vision_encoder.onnx"), sess_options=vis_opts, providers=providers
        ),
        "embedding": ort.InferenceSession(
            str(model_dir / "embedding.onnx"), sess_options=opts, providers=providers
        ),
        "llm": ort.InferenceSession(
            str(llm_path), sess_options=opts, providers=providers
        ),
    }
    print(f"Active providers: {sessions['llm'].get_providers()}\n")

    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")
    print(f"Prompt: {args.prompt}\n")

    result = generate(
        sessions, processor, config, image,
        prompt_text=args.prompt,
        max_new_tokens=args.max_tokens,
    )

    print(f"\n{'='*60}")
    print("Output:")
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
