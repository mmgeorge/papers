"""Run GLM-OCR on formula images.

Uses the official "Formula Recognition:" prompt.

Usage:
    uv run python run_formulas.py [formulas_dir]
    uv run python run_formulas.py ../../../../test-extract/formulas
    uv run python run_formulas.py --device dml ../../../../test-extract/formulas
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

DEFAULT_MODEL_DIR = "../model"
# Official GLM-OCR prompt for formula recognition (from config.yaml task_prompt_mapping)
PROMPT = "Formula Recognition:"


def generate_formula(sessions, processor, config, image, float_dtype):
    """Run generation for a single formula image. Returns LaTeX string."""
    image_token_id = config.image_token_id
    eos_token_ids = config.text_config.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].numpy()
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    image_grid_thw = inputs["image_grid_thw"].numpy().astype(np.int64)

    batch_size, seq_len = input_ids.shape

    # Build 3D position IDs (M-RoPE)
    position_ids = build_position_ids(input_ids, image_grid_thw, image_token_id)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)

    # Vision encoder
    vis_pos_ids, max_grid_size = compute_vision_pos_ids(image_grid_thw)
    vis_dtype = detect_dtype(sessions["vision_encoder"])
    vision_output = sessions["vision_encoder"].run(None, {
        "pixel_values": pixel_values.astype(vis_dtype),
        "pos_ids": vis_pos_ids.astype(np.int64),
        "max_grid_size": np.array(max_grid_size, dtype=np.int64),
    })[0]

    # Token embedding
    token_embeds = sessions["embedding"].run(
        None, {"input_ids": input_ids.astype(np.int64)}
    )[0]

    # Merge vision embeddings
    inputs_embeds = token_embeds.copy()
    image_mask = (input_ids == image_token_id)
    vision_flat = vision_output.reshape(-1, vision_output.shape[-1])
    inputs_embeds[image_mask] = vision_flat.astype(inputs_embeds.dtype)

    # Prefill
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

    # Decode loop
    generated_tokens = []
    current_pos = position_ids[:, :, -1:] + 1

    for step in range(8192):
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
        n_gen = len(generated_tokens)
        if n_gen >= 10:
            last_10 = generated_tokens[-10:]
            if all(t == last_10[-1] for t in last_10):
                break

    return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Run GLM-OCR on formula images")
    parser.add_argument("formulas_dir", nargs="?", default="../../../../test-extract/formulas",
                        help="Directory of formula images")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "dml", "cpu"])
    args = parser.parse_args()

    formulas_dir = Path(args.formulas_dir)
    image_paths = sorted(formulas_dir.glob("*.png"))
    if not image_paths:
        print(f"No PNG files found in {formulas_dir}", file=sys.stderr)
        sys.exit(1)

    model_dir = Path(args.model_dir)
    required = ["vision_encoder.onnx", "embedding.onnx", "llm.onnx"]
    missing = [f for f in required if not (model_dir / f).exists()]
    if missing:
        print(f"Missing files in {model_dir}: {', '.join(missing)}", file=sys.stderr)
        print("Run export.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading processor from {model_dir} ...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    config = AutoConfig.from_pretrained(str(model_dir))

    import io, contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        providers, opts = resolve_device(args.device)
    print(f.getvalue(), end="", file=sys.stderr)

    # Vision encoder needs ORT_ENABLE_EXTENDED
    vis_opts = ort.SessionOptions()
    vis_opts.log_severity_level = 3
    vis_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    if hasattr(opts, 'execution_mode'):
        vis_opts.execution_mode = opts.execution_mode
    if hasattr(opts, 'enable_mem_pattern'):
        vis_opts.enable_mem_pattern = opts.enable_mem_pattern

    print("Loading ONNX sessions ...", file=sys.stderr)
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

    float_dtype = detect_dtype(sessions["llm"])
    print(f"LLM dtype: {float_dtype}", file=sys.stderr)
    print(f"Prompt: {PROMPT}", file=sys.stderr)

    # Warmup
    print("Warmup...", file=sys.stderr)
    img = Image.open(str(image_paths[0])).convert("RGB")
    _ = generate_formula(sessions, processor, config, img, float_dtype)

    print(f"Running {len(image_paths)} formulas...", file=sys.stderr)
    t0 = time.perf_counter()

    for img_path in image_paths:
        image = Image.open(str(img_path)).convert("RGB")
        latex = generate_formula(sessions, processor, config, image, float_dtype)
        # Strip any surrounding $$ or $ that the model might add
        latex = latex.strip()
        if latex.startswith("$$") and latex.endswith("$$"):
            latex = latex[2:-2].strip()
        elif latex.startswith("$") and latex.endswith("$"):
            latex = latex[1:-1].strip()
        print(f"{img_path.name}\t{latex}")

    elapsed = time.perf_counter() - t0
    print(f"Done: {len(image_paths)} formulas in {elapsed:.1f}s "
          f"({elapsed/len(image_paths):.2f}s/formula)", file=sys.stderr)


if __name__ == "__main__":
    main()
