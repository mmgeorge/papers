"""ONNX Runtime inference for GLM-OCR (zai-org/GLM-OCR).

Uses 3-part ONNX model (FP16 or BF16):
  - vision_encoder.onnx  (FP16 or BF16 with FP16 Conv islands)
  - embedding.onnx       (FP16 or BF16, exported via export.py)
  - llm.onnx             (FP16 or BF16, exported via export.py)

Auto-detects dtype from model files. BF16 models require CUDA.

Usage:
  uv run python run.py --image path/to/image.png
  uv run python run.py --device cuda
  uv run python run.py --prompt "Text Recognition:"
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
    resolve_device,
    compute_vision_pos_ids,
    build_position_ids,
    detect_dtype,
)

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "../model"
DEFAULT_PROMPT = "Parse the algorithm in the image as LaTeX."
DEFAULT_MAX_TOKENS = 8192


# ── Inference ───────────────────────────────────────────────────────────

def generate(
    sessions: dict,
    processor,
    config,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = True,
):
    """Run full generation: preprocess -> vision -> embed -> prefill -> decode loop."""
    image_token_id = config.image_token_id
    eos_token_ids = config.text_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    # 1. Preprocess with the official processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].numpy()
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    image_grid_thw = inputs["image_grid_thw"].numpy().astype(np.int64)

    batch_size, seq_len = input_ids.shape
    print(f"  input_ids: {input_ids.shape}, pixel_values: {pixel_values.shape}, "
          f"grid_thw: {image_grid_thw.tolist()}")

    # 2. Build 3D position IDs (M-RoPE)
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
    print(f"  Vision: {vision_output.shape} in {t_vision:.3f}s")

    # 4. Token embedding
    token_embeds = sessions["embedding"].run(
        None, {"input_ids": input_ids.astype(np.int64)}
    )[0]

    # 5. Merge: replace image token positions with vision embeddings
    inputs_embeds = token_embeds.copy()
    image_mask = (input_ids == image_token_id)
    vision_flat = vision_output.reshape(-1, vision_output.shape[-1])
    inputs_embeds[image_mask] = vision_flat.astype(inputs_embeds.dtype)

    # Detect LLM dtype from ONNX model inputs
    float_dtype = detect_dtype(sessions["llm"])

    # 6. Prefill: run LLM with full sequence, empty KV cache
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
    print(f"  Prefill: {t_prefill:.3f}s")

    # 7. Decode loop
    t0 = time.perf_counter()
    generated_tokens = []
    current_pos = position_ids[:, :, -1:] + 1

    for step in range(max_new_tokens):
        # Cast to float32 for argmax stability (FP16/BF16 not supported by numpy argmax)
        next_token = int(np.argmax(logits[:, -1, :].astype(np.float32), axis=-1)[0])
        generated_tokens.append(next_token)

        if next_token in eos_token_ids:
            break

        if stream:
            token_str = processor.tokenizer.decode([next_token], skip_special_tokens=False)
            try:
                print(token_str, end="", flush=True)
            except UnicodeEncodeError:
                print("?", end="", flush=True)

        # Embed next token
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
        n_gen = len(generated_tokens)
        if n_gen >= 10:
            last_10 = generated_tokens[-10:]
            if all(t == last_10[-1] for t in last_10):
                if stream:
                    print(f"\n  [repetition detected at step {step}]", flush=True)
                break

    t_decode = time.perf_counter() - t0
    num_tokens = len(generated_tokens)
    tok_per_sec = (num_tokens - 1) / t_decode if t_decode > 0 and num_tokens > 1 else 0

    if stream:
        print()
    print(f"\n--- Stats ---")
    print(f"Tokens: {num_tokens}")
    print(f"Prefill: {t_prefill:.3f}s")
    print(f"Decode: {t_decode:.2f}s ({tok_per_sec:.1f} tok/s)")
    print(f"Total: {t_prefill + t_decode:.2f}s")

    output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GLM-OCR ONNX inference")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "dml", "cpu"])
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Check files exist
    required = ["vision_encoder.onnx", "embedding.onnx", "llm.onnx"]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        print(f"Missing files in {model_dir}: {', '.join(missing)}")
        print("Run export.py first to generate the ONNX models.")
        sys.exit(1)

    # Load processor and config
    print(f"Loading processor from {model_dir} ...")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    config = AutoConfig.from_pretrained(str(model_dir))

    # Load ONNX sessions
    providers, opts = resolve_device(args.device)

    # Vision encoder needs ORT_ENABLE_EXTENDED (not ENABLE_ALL) because
    # the FP16 conversion blocks Cast-to-FLOAT nodes, creating a mixed-type
    # graph that ORT's SimplifiedLayerNormFusion can't handle.
    vis_opts = ort.SessionOptions()
    vis_opts.log_severity_level = 3
    vis_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if hasattr(opts, 'execution_mode'):
        vis_opts.execution_mode = opts.execution_mode
    if hasattr(opts, 'enable_mem_pattern'):
        vis_opts.enable_mem_pattern = opts.enable_mem_pattern

    print("Loading ONNX sessions ...")
    sessions = {
        "vision_encoder": ort.InferenceSession(
            str(model_dir / "vision_encoder.onnx"), sess_options=vis_opts, providers=providers
        ),
        "embedding": ort.InferenceSession(
            str(model_dir / "embedding.onnx"), sess_options=opts, providers=providers
        ),
        "llm": ort.InferenceSession(
            str(model_dir / "llm.onnx"), sess_options=opts, providers=providers
        ),
    }
    print("Sessions loaded.\n")

    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")
    print(f"Prompt: {args.prompt}")
    print()

    # Generate
    result = generate(
        sessions, processor, config, image,
        prompt_text=args.prompt,
        max_new_tokens=args.max_tokens,
        stream=True,
    )

    print(f"\n{'='*60}")
    print("Output:")
    print("="*60)
    print(result)
    print("="*60)


if __name__ == "__main__":
    main()
