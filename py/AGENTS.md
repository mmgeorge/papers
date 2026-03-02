# py/ — Python Projects

## Structure

- `pp-formulanet/` — shared library: PyTorch model definitions (encoder, decoder), weight extraction, preprocessing, shared FP32 ONNX export logic
- `pp-formulanet/cuda/` — uv project for CUDA: optimized FP16 export + CUDA graph inference (needs `onnxruntime-gpu`)
- `pp-formulanet/directml/` — uv project for DirectML: FP32 export + session.run inference (needs `onnxruntime-directml`)

## Rules

- Always use `uv` for dependency management.
- CUDA and DirectML have separate venvs because `onnxruntime-gpu` and `onnxruntime-directml` are mutually exclusive packages.
- `porting.md` documents lessons learned from porting PP-FormulaNet Plus-L from PaddleOCR to optimized ONNX.
