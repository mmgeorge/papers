# Porting PP-FormulaNet from Python to Rust (ort + cudarc)

This documents the complete process of porting the PP-FormulaNet CUDA inference
pipeline from Python (onnxruntime + ctypes cudart) to Rust (ort 2.0.0-rc.11 +
cudarc 0.16). The model is a split encoder/decoder transformer that converts
formula images to LaTeX.

## High-Level Porting Steps

### 1. Export the ONNX models

Run `py/pp-formulanet/cuda/export.py` to produce:
- `encoder_fp16.onnx` — ViT encoder, FP16 weights and I/O
- `decoder_fp16_argmax.onnx` — autoregressive decoder with GPU-side ArgMax appended

The ArgMax node avoids transferring 100KB logits (50000 × f16) to CPU each step;
instead only an 8-byte int64 `next_token` crosses the PCIe bus.

Place both in the model cache dir (`{dirs::cache_dir()}/papers/models/`).

### 2. Set up ORT with CUDA EP and dynamic loading

```toml
ort = { version = "=2.0.0-rc.11", features = ["cuda", "load-dynamic", "half"] }
cudarc = { version = "0.16", default-features = false, features = ["std", "driver", "dynamic-linking", "cuda-version-from-build-system"] }
```

ORT is loaded dynamically at runtime (`load-dynamic` feature) from a local
`onnxruntime/lib/onnxruntime.dll`. cudarc also dynamically links to the CUDA
driver, avoiding static CRT conflicts with other native dependencies.

### 3. Create a shared CUDA stream between cudarc and ORT

```rust
let cuda_ctx = cudarc::driver::CudaContext::new(0)?;  // shares primary context with ORT
let cuda_stream = cuda_ctx.new_stream()?;  // NOT default_stream() — see below
let raw_stream = cuda_stream.cu_stream() as *mut ();

let decoder_ep = unsafe {
    CUDAExecutionProvider::default()
        .with_cuda_graph(false)   // disabled — we capture our own (see §12)
        .with_compute_stream(raw_stream)
        .build()
};
```

This is critical: ORT and cudarc must operate on the **same CUDA stream** so that
async memcpy operations are ordered relative to `run_binding()` kernel launches
via stream semantics — no explicit synchronization needed between them.

We use `new_stream()` instead of `default_stream()` because the CUDA default
stream (stream 0) does not support `cudaStreamBeginCapture`, which we need for
external CUDA graph capture (§12).

### 4. Pre-allocate ALL decoder GPU buffers at init

Use ort's `Allocator` to allocate fixed GPU tensors that persist for the lifetime
of the predictor:

```rust
let allocator = Allocator::new(&session, cuda_mem.clone())?;
let input_ids = Tensor::<i64>::new(&allocator, [1, 1])?;
let step = Tensor::<i64>::new(&allocator, [1])?;
let enc_hidden = Tensor::<f16>::new(&allocator, enc_output_shape)?;
let kv_cache = Tensor::<f16>::new(&allocator, [1, N_HEADS, MAX_SEQ, HEAD_DIM])?;  // × 16
let logits = Tensor::<f16>::new(&allocator, [1, 1, 50000])?;
let next_token = Tensor::<i64>::new(&allocator, [1])?;
```

Save raw GPU pointers (`data_ptr_mut() as u64`) before converting to `Value`,
since `Value` (a type-erased DynValue) doesn't expose `data_ptr_mut()`.

### 5. Create ONE persistent IoBinding with in-place KV cache

```rust
let mut binding = session.create_binding()?;
binding.bind_input("input_ids", &input_ids_val)?;   // borrows
binding.bind_input("step", &step_val)?;              // borrows
binding.bind_input("encoder_hidden_states", &enc)?;  // borrows
for kv in &kv_vals { binding.bind_input(&past_name, kv)?; }

// In-place KV: bind same buffer as output via raw C API
for kv in &kv_vals { raw_bind_output(&mut binding, &present_name, kv)?; }

binding.bind_output("logits", logits)?;              // moves
binding.bind_output("next_token", next_token)?;      // moves
```

The binding is **never recreated** — reused across all formulas and all steps.
CUDA graphs capture buffer addresses on first `run_binding()` and replay the
exact same kernel sequence on subsequent calls.

### 6. Decoder loop with batched sync (K=4)

```
Per formula:
  1. D2D: encoder output → fixed enc_hidden buffer
  2. memset: zero all 16 KV cache buffers
  3. H2D: seed input_ids with BOS token

  For each batch of K=4 steps:
    For each step in batch:
      a. H2D: step counter (8 bytes, async)
      b. run_binding() — CUDA graph replay
      c. D2D: next_token → input_ids (8 bytes, GPU-only)
      d. D2H: next_token → CPU buffer slot (8 bytes, async)
    sync once
    scan K tokens for EOS, return if found
```

### 7. Warmup and graph capture

Run 3 full formula predictions (encode + decode) during init to:
- Prime ORT's internal memory pools and kernel caches
- Trigger ORT's CUDA kernel JIT compilation
- Establish stable GPU clock frequencies

After warmup, capture a single decoder step into a cudarc `CudaGraph` via
`cudaStreamBeginCapture`/`cudaStreamEndCapture` (see §12 for details). All
subsequent decoder steps replay this captured graph.

---

## Key Differences: Python vs Rust

| Aspect | Python | Rust |
|--------|--------|------|
| CUDA graphs | External via ctypes `cudart.cudaGraphLaunch` | External via cudarc `CudaGraph::launch()` (§12) |
| ORT graph setting | `enable_cuda_graph: "1"` (but not used for replay) | `.with_cuda_graph(false)` (disabled, capture our own) |
| IoBinding | New `io_binding()` per step | One persistent binding for all steps/formulas |
| KV cache | `bind_ortvalue_output` takes reference → in-place | Raw C API `BindOutput` → in-place (bypasses ownership) |
| KV reset | Re-allocate from numpy zeros each formula | `memset_d8_async` (keep addresses fixed for CUDA graphs) |
| Token feed | H2D `cudaMemcpy` from CPU each step | D2D `memcpy_dtod_async` on GPU (no PCIe) |
| Sync | Implicit via synchronous `cudaMemcpy` each step | Batched: async D2H + one `stream.synchronize()` per K steps |
| Encoder output | `OrtValue.ortvalue_from_numpy` to GPU | D2D copy from encoder output to fixed buffer |
| Vocab size | Implicit from model | Must use model's 50000, not tokenizer's 49292 |

## Performance Results (vbd.pdf, 151 formulas)

| Version | Per-formula | Total | Speedup |
|---------|-------------|-------|---------|
| Original (per-step IoBinding, no CUDA graphs) | ~205ms | ~31s | 1.0x |
| Persistent IoBinding + ORT CUDA graphs | ~96ms | ~14.5s | 2.1x |
| + In-place KV (eliminated D2D KV copies) | ~96ms | ~14.5s | 2.1x |
| + Batched sync K=4 + GPU token feed | ~83ms | ~12.5s | 2.5x |
| + External CUDA graph (bypass ORT sync) | ~71ms | ~10.8s | 2.9x |
| + Pinned host memory (async D2H) | **~71ms** | **~10.8s** | **2.9x** |
| Python reference | ~71ms | ~10.7s | 2.9x |

---

## Detailed Issues and Fixes

### 1. CUDA graphs require fixed buffer addresses

**Symptom:** `STATUS_ACCESS_VIOLATION` crash on step 1 (step 0 works).

**Cause:** CUDA graphs capture the exact GPU buffer addresses during the first
`run_binding()` call. On subsequent calls, ORT replays the captured graph which
references those specific addresses. If you create a new `IoBinding` per step,
ORT allocates new buffers at different addresses — the replayed graph reads/writes
to the old (now-freed) addresses.

**Fix:** Create ONE `IoBinding` at init with pre-allocated buffers. Never drop or
recreate it. All subsequent `run_binding()` calls replay the graph against the
same fixed addresses.

**Python note:** Python creates a new `io_binding()` per step but reuses the same
`OrtValue` objects, so buffer addresses stay the same. The new Python binding
object is lightweight — it just re-registers the same pointers.

### 2. Allocator must outlive all Values

**Symptom:** `STATUS_ACCESS_VIOLATION` crash at step 0, "typeinfo_ptr to not be
null" error.

**Cause:** `Allocator::new(&session, mem_info)` returns an `Allocator` that owns
the underlying ORT allocator handle. All `Tensor::new(&allocator, shape)` calls
allocate GPU memory through this handle. When the `Allocator` is dropped, ALL
memory allocated through it is freed — even if `Value` objects still reference it.

If `Allocator` is a local variable in a builder function, it's dropped when the
function returns, leaving all Values with dangling GPU pointers.

**Fix:** Store `_allocator: Allocator` as the **last field** in the struct. Rust
drops struct fields in declaration order, so the allocator is freed after the
binding and all Values.

```rust
struct DecoderState {
    session: Session,
    binding: IoBinding,     // dropped first
    _input_ids: Value,      // dropped second
    _kv: Vec<Value>,        // dropped third
    _allocator: Allocator,  // dropped LAST — frees GPU memory
}
```

### 3. ort's bind_output moves the Value (no in-place KV)

**Symptom:** Can't bind the same buffer as both input and output in Rust.

**Cause:** `binding.bind_input(name, &value)` borrows `&Value`, but
`binding.bind_output(name, value)` takes `Value` by value (moves it). You can't
pass the same `Value` to both — the first `bind_output` consumes it.

There is a `bind_output_mut(&mut Value)` method in ort, but it's `pub(crate)` —
not accessible from outside the crate. `Value::clone()` does a **deep copy**
(allocates a new GPU buffer and copies data), not a shared reference.

**Fix:** Call the ORT C API `BindOutput` directly, bypassing Rust's ownership:

```rust
use ort::AsPointer;
let api = ort::api();
let status = unsafe {
    (api.BindOutput)(
        binding.ptr() as *mut ort::sys::OrtIoBinding,
        c_name.as_ptr(),
        kv_val.ptr() as *mut ort::sys::OrtValue,
    )
};
if !status.0.is_null() {
    unsafe { (api.ReleaseStatus)(status.0) };
    return Err(/* ... */);
}
```

The `Value` must be kept alive in the struct (it's still bound as input via
`bind_input`, which holds an `Arc` clone of the inner pointer). ORT reads from
and writes to the same GPU buffer in-place.

**Python note:** Python's `io.bind_ortvalue_output(name, ort_value)` takes a
reference (Python has no ownership semantics), so in-place KV is trivial.

### 4. Wrong logits vocab_size

**Symptom:** "Got invalid dimensions for output: logits - index: 2 Got: 49292
Expected: 50000"

**Cause:** The tokenizer reports `vocab_size = 49292`, but the ONNX model's logits
output shape is `[1, 1, 50000]` (padded for alignment). Pre-allocating the logits
buffer with the tokenizer's size causes a shape mismatch.

**Fix:** Always use the model's output shape (50000), not the tokenizer's vocab
size. The extra entries are never selected by ArgMax.

### 5. Getting GPU pointers from DynValue

**Symptom:** `data_ptr_mut()` is only available on typed `Tensor<T>`, not on
`Value` (which is `DynValue`).

**Cause:** After converting `Tensor<T>` to `Value` via `.into()`, the typed
interface is lost. But we need raw GPU pointers for cudarc memcpy operations.

**Fix:** Save pointers before conversion, OR use the ORT C API directly:

```rust
// Option A: Save before conversion
let mut tensor = Tensor::<i64>::new(&allocator, [1, 1])?;
let ptr = tensor.data_ptr_mut() as u64;  // save pointer
let value: Value = tensor.into();         // now it's DynValue

// Option B: Extract from DynValue via C API
fn value_data_ptr(value: &Value) -> u64 {
    use ort::AsPointer;
    let mut ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        (ort::api().GetTensorMutableData)(
            value.ptr() as *mut OrtValue,
            &mut ptr as *mut *mut c_void as *mut *mut _,
        );
    }
    ptr as u64
}
```

### 6. Encoder output shape is not statically known

**Symptom:** Need to pre-allocate `enc_hidden` buffer but don't know the encoder
output dimensions.

**Cause:** The encoder output shape depends on the model architecture (e.g.,
`[1, 2304, 512]` for a ViT with 768×768 input). This isn't available in ONNX
metadata (dimensions are listed as dynamic).

**Fix:** Run a dummy forward pass through a temporary encoder session at init time:

```rust
let dummy_image = preprocess_image(&DynamicImage::new_rgb8(64, 64));
let outputs = probe_session.run_binding(&binding)?;
let shape = match outputs[0].dtype() {
    ValueType::Tensor { shape, .. } => shape,
    _ => panic!("not a tensor"),
};
```

The probe session is dropped after extracting the shape. All real images are
preprocessed to 768×768, so the shape is deterministic.

### 7. Stream synchronization overhead dominates

**Symptom:** Nsight Systems profiling shows `cudaStreamSynchronize` accounts for
52% of total GPU API time (~722µs average per call, ~5.2s total across 151
formulas).

**Cause:** The autoregressive decoder must check each token for EOS (end of
sequence) to know when to stop. This requires reading the 8-byte `next_token`
from GPU to CPU, which requires a stream sync. With ~1,500 total decoder steps,
that's 1,500 sync calls.

**Fix:** Batched sync — run K=4 steps without syncing, enqueue async D2H into
separate CPU buffer slots, sync once, then scan all K tokens for EOS:

```rust
for k in 0..BATCH_K {
    // H2D step, run_binding, D2D next→input, async D2H next→buf[k]
}
stream.synchronize();  // one sync for K steps
for k in 0..BATCH_K {
    if token_buf[k] == EOS_ID { return; }
}
```

This cuts sync calls by ~75%. Steps after EOS produce garbage tokens that are
discarded. At most K-1 wasted steps per formula (~2ms vs ~500ms sync savings).

### 8. Token feeding: avoid CPU round-trip

**Symptom:** Each step does GPU→CPU (read token) then CPU→GPU (write as next
input), crossing PCIe twice for 8 bytes.

**Cause:** The Python reference uses synchronous `cudaMemcpy` for both H2D and
D2H each step. In the original Rust port, we did the same.

**Fix:** Feed the token back to the model entirely on GPU via D2D copy:

```rust
// After run_binding produces next_token on GPU:
cudarc::driver::result::memcpy_dtod_async(
    input_ids_ptr,    // destination: model's input buffer
    next_token_ptr,   // source: model's output buffer
    8,                // one i64
    cu_stream,
);
```

The D2H to CPU is still needed for EOS checking, but the input feeding no longer
crosses PCIe. Only the step counter (which increments on CPU) needs H2D.

### 9. cudarc memcpy_dtoh_sync vs async+sync

**Symptom:** Trying to simplify by using synchronous `memcpy_dtoh_sync` instead
of `memcpy_dtoh_async` + `stream.synchronize()`.

**Cause:** `cuMemcpyDtoH_v2` (synchronous) synchronizes the **entire device**,
not just our stream. This is heavier than `cuStreamSynchronize` which only waits
on the specific stream we share with ORT.

**Measured:** Synchronous D2H was consistently ~1-2s slower across 5 runs.

**Fix:** Keep using `memcpy_dtoh_async` + `stream.synchronize()` — it only blocks
on our stream, allowing other GPU work (if any) to continue.

### 10. Unnecessary stream synchronization

**Symptom:** Three `stream.synchronize()` calls per step that aren't needed.

**Cause:** When ORT and cudarc share the same CUDA stream via
`with_compute_stream`, all operations are **ordered by stream semantics**. An
async memcpy followed by `run_binding()` on the same stream guarantees the
memcpy completes before any kernels start — no explicit sync needed.

**Fix:** Remove all syncs except after D2H for reading `next_token` on CPU:
- ~~sync after memset KV~~ — `run_binding` on same stream waits automatically
- ~~sync after H2D input_ids/step~~ — `run_binding` on same stream waits
- ~~sync after D2D kv copies~~ — next step's operations on same stream wait
- **Keep:** sync after D2H `next_token` — CPU needs the value

### 11. tokenizers crate CRT conflict

**Symptom:** Linker error: conflicting `/MT` (static CRT) vs `/MD` (dynamic CRT).

**Cause:** The `tokenizers` crate's default `esaxx-rs` feature compiles C++ code
with static CRT (`/MT`), conflicting with `clipper2c-sys` which uses dynamic CRT
(`/MD`). Windows requires all object files in a binary to use the same CRT.

**Fix:** Disable `esaxx_fast` (only needed for BPE training, not inference):

```toml
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
```

### 12. ORT's CUDA graph replay forces unnecessary synchronization

**Symptom:** Rust with ORT's built-in CUDA graphs runs at ~83ms/formula vs Python's
~71ms/formula, despite identical model, ONNX files, and CUDA graph usage. nsys
profiling shows 3,744 `cudaStreamSynchronize` calls totaling 5.6s (57.9% of total
GPU API time).

**Cause:** ORT's `CUDAGraphManager::Replay()` in `cuda_graph.cc` unconditionally
calls `cudaStreamSynchronize` after every `cudaGraphLaunch()`:

```cpp
// onnxruntime/core/providers/cuda/cuda_graph.cc
Status CUDAGraphManager::Replay(cudaStream_t stream) {
  CUDA_CALL_THROW(cudaGraphLaunch(graph_exec_, stream));
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));  // <-- hardcoded, no API to disable
  return Status::OK();
}
```

This is called from `CUDAExecutionProvider::PerThreadContext::ReplayGraph()` with
`sync_status_flag=true` hardcoded. There is no public API, session option, or
environment variable to skip this sync. The ort Rust crate's `run_binding()` adds
no overhead — it's a thin wrapper around ORT's C `RunWithBinding` API.

Python avoids this because its CUDA graph capture/replay is done externally via
ctypes `cudart.cudaGraphLaunch()`, bypassing ORT's `CUDAGraphManager` entirely.

**Fix:** Disable ORT's built-in CUDA graphs and capture our own via cudarc:

1. **Disable ORT's graphs:** `.with_cuda_graph(false)` on the decoder EP.

2. **Disable ORT's OnRunEnd sync:** `RunOptions::disable_device_sync()` sets
   `disable_synchronize_execution_providers=1`, preventing ORT from calling
   `cudaStreamSynchronize` in its `OnRunEnd` callback. This is critical both for
   avoiding unwanted sync during normal execution and for not breaking stream capture
   (sync during capture causes `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`).

3. **Use a non-default stream:** `cuda_ctx.new_stream()` instead of
   `default_stream()`. The CUDA default stream (stream 0) does not support
   `cudaStreamBeginCapture` and returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.

4. **Capture during warmup:** After 3 normal warmup inferences (to trigger ORT's
   memory allocations and kernel JIT), capture one decoder step:

   ```rust
   cuda_stream.synchronize()?;  // ensure all prior work is done
   cuda_stream.begin_capture(CU_STREAM_CAPTURE_MODE_GLOBAL)?;
   session.run_binding_with_options(&binding, &run_options)?;  // GPU work captured, not executed
   let graph = cuda_stream.end_capture(CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?;
   ```

5. **Replay in decode loop:** Replace `session.run_binding()` with `graph.launch()`.
   No forced sync per step. The only sync is our batched K=4 sync for reading tokens.

**Result:** `cudaStreamSynchronize` dropped from 3,744 to 916 calls, sync time from
5.6s to 0.24s (96% reduction). Performance: 83ms → 71.4ms/formula, matching Python.

**Verification:** External graph capture produces token-for-token identical output to
the non-graph path (tested across all 151 formulas). The 8/151 Rust-vs-Python
differences are pre-existing (image crate vs PIL preprocessing), not a regression.

### 13. cuMemcpyDtoHAsync blocks on pageable host memory

**Symptom:** After fixing §12, nsys shows `cuMemcpyDtoHAsync` as the new dominant
cost at 57% of GPU API time (5.4s total, ~840µs average per call). An 8-byte async
copy should complete in microseconds.

**Cause:** From the CUDA documentation:

> For transfers from device memory to **pageable** host memory,
> `cuMemcpyDtoHAsync` will return only once the copy has completed.

Our token buffer was stack-allocated pageable memory:

```rust
let mut token_buf = [0i64; BATCH_K];  // pageable — forces sync D2H
```

Despite the "Async" name, CUDA degrades this to a synchronous copy: the driver
waits for all prior stream work (the `graph.launch()`) to finish, performs the
copy, then returns. This meant our K=4 batched sync was illusory — every D2H
call was already an implicit sync point, stalling the CPU for ~840µs per step.

**Fix:** Allocate a **pinned (page-locked)** host buffer via `cuMemAllocHost`:

```rust
struct PinnedTokenBuf {
    ptr: *mut i64,
    len: usize,
}

impl PinnedTokenBuf {
    fn new(len: usize) -> Result<Self, ExtractError> {
        let raw = unsafe {
            cudarc::driver::result::malloc_host(len * size_of::<i64>(), 0)
        }?;
        Ok(Self { ptr: raw as *mut i64, len })
    }
}
```

We use `flags=0` (not `CU_MEMHOSTALLOC_WRITECOMBINED`) because write-combined
memory is optimized for CPU→GPU writes but slow for CPU reads — and we read the
tokens on CPU after sync.

**Result (nsys):**

| Metric | Pageable | Pinned |
|---|---|---|
| `cuMemcpyDtoHAsync` | 5.4s (840µs avg) | **27ms (4µs avg)** |
| `cuStreamSynchronize` | 7ms | 5.5s (3.4ms avg) |

The D2H is now truly async (99.5% time reduction). The total wall time is
unchanged (~71ms/formula) because the GPU compute time is the real floor — we
just shifted the wait from "implicit sync inside every D2H" to "one explicit
sync per K=4 batch."

**Why it matters despite no throughput change:** With pageable memory, the CPU
was stalled inside `cuMemcpyDtoHAsync` for ~840µs on every decoder step — ~5.4s
of CPU time burned doing nothing across 151 formulas. With pinned memory, the
CPU fires off all K=4 steps (H2D + graph launch + D2D + D2H) in microseconds,
then blocks once at `stream.synchronize()`. The CPU is free to do other work
(PDF parsing, layout detection, text extraction) while the GPU runs decoder
steps, making the formula predictor a GPU hog instead of a CPU hog.

---

## Benchmarking

A standalone formula benchmark binary isolates prediction time from the full
pipeline (PDF loading, layout detection, text extraction, table recognition):

```sh
# Dump formula crops from a PDF
papers-extract data/vbd.pdf -o test-extract/ --dump-formulas

# Benchmark formula prediction only
bench_formulas test-extract/formulas/ --runs 5
```

For CUDA profiling with Nsight Systems:

```sh
nsys profile --trace=cuda,nvtx -o formula_profile \
    bench_formulas test-extract/formulas/ --runs 1

nsys stats formula_profile.nsys-rep --report cuda_api_sum
nsys stats formula_profile.nsys-rep --report cuda_gpu_kern_sum
```

Key things to look for in the profile:
- `cuStreamSynchronize` — batched K=4 sync; should dominate (this is where we wait for GPU work)
- `cudaEventSynchronize` — ORT-internal event waits
- `cuGraphLaunch` — CUDA graph replays (should dominate over `cudaLaunchKernel`)
- `cuMemcpyDtoHAsync` — should be ~6K calls at ~4µs each (truly async with pinned memory)
- If `cuMemcpyDtoHAsync` shows ~800µs+ avg, the token buffer is pageable, not pinned (§13)
- `cuMemcpyHtoDAsync` — step counter updates
- Gap between GPU kernels — indicates pipeline stalls from sync

---

## Dependencies

```toml
ort = { version = "=2.0.0-rc.11", features = ["cuda", "load-dynamic", "half"] }
cudarc = { version = "0.16", default-features = false, features = [
    "std", "driver", "dynamic-linking", "cuda-version-from-build-system"
] }
half = "2"           # f16 type for FP16 tensor data
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
ndarray = "0.17"     # tensor construction for encoder input
image = "0.25"       # image preprocessing
```

## Files

| File | Purpose |
|------|---------|
| `src/formula.rs` | `FormulaPredictor` — encoder/decoder with CUDA graphs |
| `src/models.rs` | Model download, ORT runtime init, EP configuration |
| `src/bin/bench_formulas.rs` | Standalone formula benchmark |
| `py/pp-formulanet/cuda/export.py` | ONNX model export pipeline |
| `py/pp-formulanet/cuda/run.py` | Python reference implementation |
| `py/pp-formulanet/common/preprocess.py` | Shared preprocessing (Python) |
