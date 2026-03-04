# Porting GLM-OCR to ONNX (FP16)

Step-by-step account of how [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) was exported to ONNX Runtime in FP16, including all problems encountered and their solutions.

## Model Overview

GLM-OCR is a 0.9B parameter vision-language model for OCR, built on:
- **Vision encoder**: CogVideoX-based (SigLIP architecture), 24 transformer layers
- **Language model**: GLM-0.5B (Glm4 architecture), 16 transformer layers
- **Connector**: MLP merger that projects vision features into the LLM embedding space

The model uses **M-RoPE** (Multi-dimensional Rotary Position Embeddings) with 3D position IDs `[temporal, height, width]` for handling both text and spatial image tokens.

Native precision: **BF16** (all weights stored as `torch.bfloat16`).

## Architecture: 3-Part Split

The model is split into 3 ONNX files for efficient inference:

```
vision_encoder.onnx   (829 MB)  - image patches -> vision embeddings
embedding.onnx        (174 MB)  - token IDs -> text embeddings
llm.onnx             (1113 MB)  - embeddings + KV cache -> logits + updated KV cache
                     ──────────
                      2116 MB total
```

**Why 3 parts instead of 1?**
- Vision encoder runs once per image; LLM runs once per token
- Embedding layer is shared between prefill and decode steps
- Keeps each ONNX graph smaller and simpler for ORT optimization

## Export Process

### Step 1: Embedding Layer (torch.onnx.export)

Straightforward — wraps `model.model.language_model.embed_tokens`:

```python
class EmbedTokensWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.language_model.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)
```

Exported with dynamic axes on `batch` and `seq_len`. The model is loaded in native BF16 and cast to FP16 before export (`model.to(torch.float16)`), so the embedding table and output are FP16.

### Step 2: LLM Decoder (torch.onnx.export)

The LLM wrapper must explicitly handle KV cache as flattened inputs/outputs because `torch.onnx.export` can't handle `DynamicCache` natively:

```python
class LLMWrapper(nn.Module):
    def forward(self, inputs_embeds, attention_mask, position_ids,
                past_key_0, past_value_0, past_key_1, past_value_1,
                ...  # 16 layers x 2 (key + value) = 32 tensors
                past_key_15, past_value_15):
        # Reconstruct DynamicCache from flat tensors
        past_key_values = DynamicCache()
        for layer in range(16):
            past_key_values.update(past_keys[layer], past_values[layer], layer)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,   # shape: [3, batch, seq_len] (M-RoPE)
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Return logits + updated KV cache as flat tuple
        logits = self.lm_head(outputs.last_hidden_state)
        return (logits, new_key_0, new_value_0, ..., new_key_15, new_value_15)
```

**Key details:**
- Uses explicit named parameters (not `*args`) for torch 2.10 compatibility with `dynamic_axes`
- `position_ids` shape is `[3, batch, seq_len]` — the M-RoPE 3D positions
- KV cache shapes: `[batch, 8_kv_heads, past_seq_len, 128_head_dim]`
- Dynamic axes on `seq_len` and `past_seq_len` allow both prefill (long sequence, empty cache) and decode (1 token, growing cache)

### Step 3: Vision Encoder (Download + FP16 Conversion)

**This was the hardest part.** The vision encoder cannot be re-exported from PyTorch because of M-RoPE:

```python
# In modeling_glm_ocr.py line ~460:
cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
hidden_states = torch.split(hidden_states, cu_seqlens.tolist(), dim=2)
#                                          ^^^^^^^^^^^^^^^^^^^^^^^^
# Data-dependent split sizes — torch.export fails with:
# GuardOnDataDependentSymNode: Could not guard on data-dependent expression
```

**Solution:** Download the pre-exported FP32 vision encoder from [ningpp/GLM-OCR](https://huggingface.co/ningpp/GLM-OCR) on HuggingFace, then convert to FP16.

#### The Cast Node Problem

Naive FP16 conversion with `onnxconverter_common` fails:

```
Type (tensor(float16)) of output arg (/rotary_pos_emb/Cast_output_0) of node
(/rotary_pos_emb/Cast) does not match expected type (tensor(float)).
```

**Root cause:** The vision encoder graph contains 244 `Cast` nodes that cast from BF16 to FLOAT32 for operations requiring FP32 precision (Conv, Range, LayerNorm's Pow/Mul/Add). When the FP16 converter runs, it changes these Cast nodes' `to` attribute from `FLOAT` to `FLOAT16`, but the downstream ops still expect `FLOAT` inputs.

**Fix:** Use `node_block_list` to prevent the converter from touching any Cast-to-FLOAT node:

```python
# Collect all Cast nodes that cast to FLOAT (type enum = 1)
block_list = []
for n in model.graph.node:
    if n.op_type == "Cast":
        for attr in n.attribute:
            if attr.name == "to" and attr.i == 1:  # TensorProto.FLOAT
                block_list.append(n.name)

# Convert with those nodes blocked
model_fp16 = float16.convert_float_to_float16(
    model,
    keep_io_types=False,
    node_block_list=block_list,  # 244 nodes stay as Cast-to-FLOAT
)
```

This preserves all the FP32 upcast paths while converting weights and computation to FP16. The resulting graph is mixed-precision: FP16 for weights/computation, with FP32 "islands" for precision-sensitive ops.

#### ORT Optimization Caveat

The mixed-type graph breaks ORT's `SimplifiedLayerNormFusion` optimization (part of `ORT_ENABLE_ALL`). At runtime, use `ORT_ENABLE_EXTENDED` for the vision encoder:

```python
vis_opts = ort.SessionOptions()
vis_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# ORT_ENABLE_BASIC and ORT_ENABLE_EXTENDED both work fine
# ORT_ENABLE_ALL crashes during SimplifiedLayerNormFusion
```

## Inference Pipeline

The inference flow:

```
1. Preprocess (AutoProcessor)
   - Image -> pixel patches [num_patches, 1176]
   - Text -> token IDs [1, seq_len]
   - Grid dimensions [num_images, 3] (temporal, height, width)

2. Vision Encoder (FP16)
   - Input: pixel_values, pos_ids, max_grid_size
   - Output: image_embeds [num_output_tokens, 1536]

3. Embedding (FP16)
   - Input: input_ids [1, seq_len]
   - Output: text_embeds [1, seq_len, 1536]

4. Merge
   - Replace image placeholder tokens in text_embeds with vision embeddings

5. Build 3D Position IDs (M-RoPE)
   - Text tokens: same position in all 3 dims
   - Image tokens: temporal=constant, height=row, width=col

6. Prefill (LLM, FP16)
   - Full sequence + empty KV cache -> logits + filled KV cache

7. Decode Loop (LLM, FP16)
   - 1 token + KV cache -> next logits + updated KV cache
   - Greedy argmax, stream tokens
```

### M-RoPE Position IDs

GLM-OCR uses 3D position IDs `[3, batch, seq_len]`:
- **Dimension 0 (temporal):** Frame index for video, constant within a frame for images
- **Dimension 1 (height):** Row position after spatial merge (merge_size=2)
- **Dimension 2 (width):** Column position after spatial merge

For text tokens, all 3 dimensions have the same incrementing position value. For image tokens, dim 0 stays constant while dims 1 and 2 encode the 2D spatial grid.

### Vision Position IDs

The vision encoder also needs M-RoPE position IDs, computed differently from the LLM positions:

```python
# For each image frame, create a 2D grid of (h, w) positions
# Apply spatial merge: reshape [h, w] -> [h//2, 2, w//2, 2] -> [h//2, w//2, 2, 2] -> flatten
hpos = np.arange(h).reshape(-1, 1).repeat(w, axis=1)
hpos = hpos.reshape(h//2, 2, w//2, 2).transpose(0, 2, 1, 3).flatten()
# Same for width positions
pos_ids = np.stack([hpos, wpos], axis=-1)  # [num_patches, 2]
```

## BF16 vs FP16 Decision

The model is natively BF16 in PyTorch — Conv2d/Conv3d genuinely compute in BF16
on GPU (PyTorch classifies them as `lower_precision_fp`, same as MatMul). Only
RMSNorm, rotary embeddings, and softmax explicitly upcast to FP32.

However, ONNX Runtime support for BF16 is limited:

| | BF16 | FP16 |
|---|---|---|
| File size | Same (2 bytes/param) | Same (2 bytes/param) |
| ORT CPU support | No | Yes |
| ORT DML support | No | Yes |
| ORT CUDA MatMul/Add/Mul | Yes | Yes |
| ORT CUDA Conv | **No** (no kernel, even opset 22) | Yes |

The ONNX spec added BF16 to Conv in opset 22 (ONNX 1.17.0), but ORT has no
BF16 Conv kernel on any provider as of v1.24 ([issue #25740](https://github.com/microsoft/onnxruntime/issues/25740)).

### Export modes

- **`--fp16`** (default): Everything in FP16. Works on CPU, DML, and CUDA.
- **`--bf16`**: Everything in BF16. Vision encoder Conv ops stay in FP16
  (ORT Conv limitation) with Cast wrappers. Requires CUDA.
- **`--fp32`**: Everything in FP32. Debug/reference only.

### BF16 Vision Encoder Conversion

The BF16 vision encoder is created by post-processing the FP16 model:

1. Convert all FP16 initializers, type annotations, and Cast nodes to BF16
2. Revert Conv weight/bias initializers back to FP16
3. Insert `Cast(BF16→FP16)` before Conv data inputs
4. Insert `Cast(FP16→BF16)` after Conv outputs

This creates a mostly-BF16 graph with FP16 islands around Conv nodes:
```
BF16 computation → Cast(BF16→FP16) → Conv(FP16) → Cast(FP16→BF16) → BF16 computation
```

Conv nodes inside FP32 islands (behind blocked Cast-to-FLOAT nodes) are
untouched — they already operate in FP32 with existing Cast wrappers.

## Dependencies

```
torch>=2.10
transformers>=5.2.0
onnx
onnxconverter-common
onnxruntime>=1.24  (or onnxruntime-directml for GPU)
numpy
ml-dtypes
Pillow
huggingface-hub
```

The `transformers 5.2.0` monkey-patch in both scripts works around a bug in `video_processing_auto.py` where `video_processor_class_from_name` fails with a `TypeError`.

## Size Comparison

| Stage | Size |
|---|---|
| Original PyTorch (BF16) | ~1.8 GB |
| FP32 ONNX (naive) | 4227 MB |
| FP16 ONNX (final) | 2116 MB |
| BF16 ONNX (embedding+LLM) | 2116 MB (same size, native precision) |

The FP16 export is 50% smaller than FP32, with identical output quality verified over 500 tokens.

## Running

```bash
# Export FP16 (default, works everywhere)
python export.py

# Export BF16 (CUDA only, native precision for embedding+LLM)
python export.py --bf16

# Inference
python run.py --image path/to/image.png
python run.py --image path/to/image.png --device cpu
python run.py --image path/to/image.png --device cuda
python run.py --image path/to/image.png --prompt "Describe this image."
```
