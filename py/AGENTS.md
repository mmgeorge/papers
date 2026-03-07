# py/ — Python Projects

## Structure

### glm-ocr/
GLM-OCR (zai-org/GLM-OCR) ONNX export and inference. Uses HuggingFace transformers model directly (no re-implementation).

Three inference backends are supported:
- **CUDA** (Windows, NVIDIA GPU): BF16 precision + CUDA graphs + GQA/MHA fusion. Uses 4 models: `vision_encoder_mha.onnx`, `embedding.onnx`, `llm.onnx`, `llm_decoder_gqa.onnx`
- **CoreML** (macOS, Apple Silicon): FP32 precision, `session.run()` decode. Uses 3 models: `vision_encoder.onnx`, `embedding.onnx`, `llm.onnx`
- **CPU** (everywhere): FP32 precision, `session.run()` decode. Same 3 models as CoreML

Directory structure:
- `common/` — shared library: model wrappers, export functions, BF16 conversion, M-RoPE position IDs, transformers monkey-patch
- `cuda/` — uv project for CUDA: ONNX export (BF16 + FP32), GQA/MHA graph surgery, inference (needs `onnxruntime-gpu`, `torch` for export)
- `ollama/` — uv project for Ollama API: formula and full-page PDF OCR via local Ollama server (minimal deps, no torch/onnx)
- `model/` — exported ONNX models + HuggingFace tokenizer/processor configs

## Rules

- Always use `uv` for dependency management.
- `porting.md` documents lessons learned from porting models to optimized ONNX.
