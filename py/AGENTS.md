# py/ — Python Projects

## Structure

### pp-formulanet/
PyTorch re-implementation of PP-FormulaNet Plus-L for ONNX export and inference.

- `common/` — shared library: model definitions (encoder, decoder), weight extraction, preprocessing, shared FP32 ONNX export logic
- `cuda/` — uv project for CUDA: optimized FP16 export + CUDA graph inference (needs `onnxruntime-gpu`)
- `directml/` — uv project for DirectML: FP32 export + session.run inference (needs `onnxruntime-directml`)

### glm-ocr/
GLM-OCR (zai-org/GLM-OCR) ONNX export and inference. Uses HuggingFace transformers model directly (no re-implementation).

- `common/` — shared library: model constants, dtype detection, device resolution, M-RoPE position ID computation, transformers monkey-patch
- `cuda/` — uv project for CUDA: ONNX export (3-part model + CUDA-graph decoder), MHA/GQA optimization, and inference (needs `onnxruntime-gpu`, `torch` for export)
- `directml/` — uv project for DirectML: MHA-optimized export + session.run inference (needs `onnxruntime-directml`)
- `ollama/` — uv project for Ollama API: formula and full-page PDF OCR via local Ollama server (minimal deps, no torch/onnx)
- `model/` — exported ONNX models + HuggingFace tokenizer/processor configs

## Rules

- Always use `uv` for dependency management.
- CUDA and DirectML have separate venvs because `onnxruntime-gpu` and `onnxruntime-directml` are mutually exclusive packages.
- `porting.md` documents lessons learned from porting models to optimized ONNX.
