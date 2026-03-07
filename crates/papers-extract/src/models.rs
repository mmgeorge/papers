use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder};

use crate::error::ExtractError;
use crate::glm_ocr::{GlmOcrConfig, GlmOcrPredictor};
use crate::tableformer::TableFormerPredictor;
use crate::TableModel;

/// HuggingFace repo for GLM-OCR ONNX models.
const GLM_OCR_REPO: &str = "mgeorge412/glm_ocr";
/// HuggingFace repo for TableFormer ONNX models.
const TABLEFORMER_REPO: &str = "mgeorge412/tableformer";

/// Model file metadata for download (layout model from GitHub releases).
struct ModelFile {
    filename: &'static str,
    url: &'static str,
}

// Layout detection (always required) — third-party, stays on GitHub releases
const LAYOUT_MODEL: ModelFile = ModelFile {
    filename: "pp-doclayoutv3.onnx",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.6.0/pp-doclayoutv3.onnx",
};

/// Resolved paths to all required model files.
pub struct ModelPaths {
    pub layout: PathBuf,
    pub glm_ocr: GlmOcrModelPaths,
    pub tableformer: Option<TableFormerModelPaths>,
}

/// Resolved paths for TableFormer models.
pub struct TableFormerModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub bbox_decoder: PathBuf,
}

/// Resolved paths for GLM-OCR models.
pub struct GlmOcrModelPaths {
    pub vision_encoder: PathBuf,
    pub embedding: PathBuf,
    pub llm: PathBuf,
    pub llm_decoder: PathBuf,
    pub tokenizer: PathBuf,
}

/// Layout model cache directory (for GitHub releases download).
/// Uses `PAPERS_MODEL_DIR` env var, `--model-cache-dir`, or platform default.
pub fn layout_cache_dir(cli_override: Option<&Path>) -> PathBuf {
    if let Some(dir) = cli_override {
        return dir.to_path_buf();
    }
    if let Ok(dir) = std::env::var("PAPERS_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("papers")
        .join("models")
}

/// Local model override directory from CLI flag or env var.
/// When set, local files are checked before downloading from HuggingFace.
fn local_model_override(cli_override: Option<&Path>) -> Option<PathBuf> {
    cli_override
        .map(|p| p.to_path_buf())
        .or_else(|| std::env::var("PAPERS_MODEL_DIR").ok().map(PathBuf::from))
}

/// Create the HuggingFace Hub API client.
fn create_hf_api() -> Result<Api, ExtractError> {
    ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(|e| ExtractError::Download(format!("HuggingFace Hub init failed: {e}")))
}

/// Ensure a model file is available, checking local override first, then HuggingFace.
fn ensure_hf_model(
    local_override: Option<&Path>,
    api: &Api,
    repo_id: &str,
    filename: &str,
) -> Result<PathBuf, ExtractError> {
    // Check local override directory first
    if let Some(dir) = local_override {
        let local_path = dir.join(filename);
        if local_path.exists() {
            return Ok(local_path);
        }
    }
    // Download from HuggingFace (cached automatically)
    let repo = api.model(repo_id.to_string());
    repo.get(filename).map_err(|e| {
        ExtractError::Download(format!("Failed to download {filename} from {repo_id}: {e}"))
    })
}

/// Ensure the ONNX external data file (.onnx.data) is co-located with the .onnx file.
/// ORT requires external data files in the same directory as the main ONNX file.
fn ensure_data_colocated(onnx_path: &Path, data_path: &Path) -> Result<(), ExtractError> {
    let onnx_dir = onnx_path.parent().unwrap();
    let data_dir = data_path.parent().unwrap();
    if onnx_dir != data_dir {
        let target = onnx_dir.join(data_path.file_name().unwrap());
        if !target.exists() {
            tracing::info!(
                "Copying {} alongside {}",
                data_path.display(),
                onnx_path.display()
            );
            std::fs::copy(data_path, &target)?;
        }
    }
    Ok(())
}

/// Download an ONNX model and its companion .onnx.data file from HuggingFace.
/// Returns the path to the .onnx file (data file is ensured co-located).
fn ensure_hf_onnx_with_data(
    local_override: Option<&Path>,
    api: &Api,
    repo_id: &str,
    filename: &str,
) -> Result<PathBuf, ExtractError> {
    let onnx_path = ensure_hf_model(local_override, api, repo_id, filename)?;
    let data_filename = format!("{filename}.data");
    // Try to download the .onnx.data file; ignore errors (not all models have external data)
    if let Ok(data_path) = ensure_hf_model(local_override, api, repo_id, &data_filename) {
        ensure_data_colocated(&onnx_path, &data_path)?;
    }
    Ok(onnx_path)
}

/// Ensure all models for the selected table engine are available.
pub fn ensure_models(
    table: TableModel,
    model_cache_dir: Option<&Path>,
) -> Result<ModelPaths, ExtractError> {
    let local_override = local_model_override(model_cache_dir);
    let api = create_hf_api()?;

    // Layout detection (always required) — downloaded from GitHub releases
    let cache_dir = layout_cache_dir(model_cache_dir);
    std::fs::create_dir_all(&cache_dir)?;
    let layout = ensure_layout_model(&cache_dir)?;

    // GLM-OCR models (always required for formula recognition)
    let glm_ocr = ensure_glm_ocr_models(local_override.as_deref(), &api)?;

    // TableFormer models (optional)
    let tableformer = if table == TableModel::TableFormer {
        Some(ensure_tableformer_models(local_override.as_deref(), &api)?)
    } else {
        None
    };

    Ok(ModelPaths {
        layout,
        glm_ocr,
        tableformer,
    })
}

/// Ensure the layout model is downloaded from GitHub releases.
pub fn ensure_layout_model(cache_dir: &Path) -> Result<PathBuf, ExtractError> {
    let local_path = cache_dir.join(LAYOUT_MODEL.filename);
    if local_path.exists() {
        return Ok(local_path);
    }
    tracing::info!(
        "Downloading {} from {}...",
        LAYOUT_MODEL.filename,
        LAYOUT_MODEL.url
    );
    download_file(LAYOUT_MODEL.url, &local_path)?;
    Ok(local_path)
}

/// Ensure all GLM-OCR model files are available.
///
/// For vision encoder: tries `vision_encoder_mha.onnx` first (CUDA/MHA-fused),
/// falls back to `vision_encoder.onnx` (CPU/CoreML FP32).
/// For decoder: `llm_decoder_gqa.onnx` is optional (non-CUDA backends don't need it).
fn ensure_glm_ocr_models(
    local_override: Option<&Path>,
    api: &Api,
) -> Result<GlmOcrModelPaths, ExtractError> {
    // Vision encoder: prefer MHA-fused, fall back to raw
    let vision_encoder =
        ensure_hf_onnx_with_data(local_override, api, GLM_OCR_REPO, "vision_encoder_mha.onnx")
            .or_else(|_| {
                ensure_hf_onnx_with_data(
                    local_override,
                    api,
                    GLM_OCR_REPO,
                    "vision_encoder.onnx",
                )
            })
            .map_err(|_| ExtractError::ModelNotFound {
                path: "vision_encoder_mha.onnx or vision_encoder.onnx".to_string(),
            })?;

    let embedding =
        ensure_hf_onnx_with_data(local_override, api, GLM_OCR_REPO, "embedding.onnx")?;
    let llm = ensure_hf_onnx_with_data(local_override, api, GLM_OCR_REPO, "llm.onnx")?;

    // Decoder: optional (non-CUDA backends don't need it)
    let llm_decoder =
        ensure_hf_onnx_with_data(local_override, api, GLM_OCR_REPO, "llm_decoder_gqa.onnx")
            .unwrap_or_else(|_| PathBuf::from("llm_decoder_gqa.onnx"));

    let tokenizer = ensure_hf_model(local_override, api, GLM_OCR_REPO, "tokenizer.json")?;

    Ok(GlmOcrModelPaths {
        vision_encoder,
        embedding,
        llm,
        llm_decoder,
        tokenizer,
    })
}

/// Ensure all TableFormer model files are available.
/// HF repo uses bare names (encoder.onnx); local override checks both bare
/// and legacy prefixed names (tableformer_encoder.onnx) for compatibility.
fn ensure_tableformer_models(
    local_override: Option<&Path>,
    api: &Api,
) -> Result<TableFormerModelPaths, ExtractError> {
    let encoder = ensure_tableformer_file(local_override, api, "encoder.onnx")?;
    let decoder = ensure_tableformer_file(local_override, api, "decoder.onnx")?;
    let bbox_decoder = ensure_tableformer_file(local_override, api, "bbox_decoder.onnx")?;
    Ok(TableFormerModelPaths {
        encoder,
        decoder,
        bbox_decoder,
    })
}

/// Try local override (bare name, then legacy tableformer_ prefix), then HF.
fn ensure_tableformer_file(
    local_override: Option<&Path>,
    api: &Api,
    bare_name: &str,
) -> Result<PathBuf, ExtractError> {
    if let Some(dir) = local_override {
        // Try bare name first (matching HF repo)
        let bare_path = dir.join(bare_name);
        if bare_path.exists() {
            return Ok(bare_path);
        }
        // Try legacy prefixed name (tableformer_encoder.onnx etc.)
        let prefixed = dir.join(format!("tableformer_{bare_name}"));
        if prefixed.exists() {
            return Ok(prefixed);
        }
    }
    // Download from HF
    let repo = api.model(TABLEFORMER_REPO.to_string());
    repo.get(bare_name).map_err(|e| {
        ExtractError::Download(format!(
            "Failed to download {bare_name} from {TABLEFORMER_REPO}: {e}"
        ))
    })
}

/// Build a TableFormer predictor from model paths.
pub fn build_tableformer_predictor(
    paths: &TableFormerModelPaths,
) -> Result<TableFormerPredictor, ExtractError> {
    TableFormerPredictor::new(&paths.encoder, &paths.decoder, &paths.bbox_decoder)
}

/// Build a GLM-OCR predictor from model paths (default: formula recognition prompt).
pub fn build_glm_ocr_predictor(
    paths: &GlmOcrModelPaths,
) -> Result<GlmOcrPredictor, ExtractError> {
    GlmOcrPredictor::new(
        &paths.vision_encoder,
        &paths.embedding,
        &paths.llm,
        &paths.llm_decoder,
        &paths.tokenizer,
    )
}

/// Build a GLM-OCR predictor with custom config (prompt).
pub fn build_glm_ocr_predictor_with_config(
    paths: &GlmOcrModelPaths,
    config: GlmOcrConfig,
) -> Result<GlmOcrPredictor, ExtractError> {
    GlmOcrPredictor::with_config(
        &paths.vision_encoder,
        &paths.embedding,
        &paths.llm,
        &paths.llm_decoder,
        &paths.tokenizer,
        config,
    )
}

/// Build a direct layout detector from the pp-doclayoutv3.onnx model.
pub fn build_layout_detector(
    model_path: &Path,
) -> Result<crate::layout::LayoutDetector, ExtractError> {
    crate::layout::LayoutDetector::new(model_path)
}

/// Download a file from a URL to a local path, following redirects.
fn download_file(url: &str, dest: &Path) -> Result<(), ExtractError> {
    use std::io::Write;

    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| ExtractError::Download(format!("Failed to create HTTP client: {e}")))?;

    let mut response = client
        .get(url)
        .send()
        .map_err(|e| ExtractError::Download(format!("Download request failed: {e}")))?;

    if !response.status().is_success() {
        return Err(ExtractError::Download(format!(
            "Download failed: HTTP {}",
            response.status()
        )));
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Write to a temp file first, then rename (atomic-ish on same filesystem)
    let tmp_path = dest.with_extension("tmp");
    let mut file = std::fs::File::create(&tmp_path)?;
    response
        .copy_to(&mut file)
        .map_err(|e| ExtractError::Download(format!("Failed to write model file: {e}")))?;
    file.flush()?;
    drop(file);

    std::fs::rename(&tmp_path, dest)?;
    tracing::info!("Downloaded {}", dest.display());
    Ok(())
}

/// Standalone GLM-OCR model loading (for binaries that don't use the full pipeline).
pub fn ensure_glm_ocr_models_standalone(
    model_cache_dir: Option<&Path>,
) -> Result<GlmOcrModelPaths, ExtractError> {
    let local_override = local_model_override(model_cache_dir);
    let api = create_hf_api()?;
    ensure_glm_ocr_models(local_override.as_deref(), &api)
}

/// Standalone TableFormer model loading (for binaries that don't use the full pipeline).
pub fn ensure_tableformer_models_standalone(
    model_cache_dir: Option<&Path>,
) -> Result<TableFormerModelPaths, ExtractError> {
    let local_override = local_model_override(model_cache_dir);
    let api = create_hf_api()?;
    ensure_tableformer_models(local_override.as_deref(), &api)
}

/// Initialize ORT with a dynamically loaded runtime library (Windows only).
///
/// Search order: `ORT_DYLIB_PATH` env var (set by `.cargo/config.toml` for
/// `cargo run`) → `onnxruntime.dll` next to the executable (dist bundle).
/// On macOS, ORT is statically linked (CoreML) so no init is needed.
pub fn init_ort_runtime() -> Result<(), ExtractError> {
    #[cfg(not(target_os = "windows"))]
    return Ok(());

    #[cfg(target_os = "windows")]
    {
        use std::sync::Once;
        static INIT: Once = Once::new();
        let mut init_err: Option<String> = None;

        INIT.call_once(|| {
            // 1. ORT_DYLIB_PATH env var (set by .cargo/config.toml for cargo run)
            // 2. Next to the exe (dist bundle)
            let path = std::env::var("ORT_DYLIB_PATH").ok().or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|e| e.parent().map(|d| d.join("onnxruntime.dll")))
                    .filter(|p| p.exists())
                    .map(|p| p.to_string_lossy().into_owned())
            });

            match path {
                Some(path) => {
                    eprintln!("Loading ORT runtime from: {path}");
                    match ort::init_from(&path) {
                        Ok(b) => {
                            b.commit();
                        }
                        Err(e) => {
                            init_err =
                                Some(format!("ORT init failed for '{path}': {e}"));
                        }
                    }
                }
                None => {
                    init_err = Some(
                        "onnxruntime.dll not found (set ORT_DYLIB_PATH or place next to exe)"
                            .into(),
                    );
                }
            }
        });

        if let Some(err) = init_err {
            Err(ExtractError::Model(err))
        } else {
            Ok(())
        }
    }
}
