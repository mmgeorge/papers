use std::path::{Path, PathBuf};

use crate::error::ExtractError;
use crate::formula::FormulaPredictor;
use crate::glm_ocr::{GlmOcrConfig, GlmOcrPredictor};
use crate::tableformer::TableFormerPredictor;
use crate::{FormulaModel, TableModel};

/// Model file metadata for download.
struct ModelFile {
    filename: &'static str,
    url: &'static str,
}

// Layout detection (always required)
const LAYOUT_MODEL: ModelFile = ModelFile {
    filename: "pp-doclayoutv3.onnx",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.6.0/pp-doclayoutv3.onnx",
};

// Formula tokenizer (required for formula recognition)
const FORMULA_TOKENIZER: ModelFile = ModelFile {
    filename: "unimernet_tokenizer.json",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/unimernet_tokenizer.json",
};

// Formula encoder (split FP16 model — expected in cache dir, no auto-download yet)
const FORMULA_ENCODER: ModelFile = ModelFile {
    filename: "encoder_fp16.onnx",
    url: "", // No auto-download — must be pre-exported via export.py
};

// Formula decoder (split FP16 model with argmax — expected in cache dir)
const FORMULA_DECODER: ModelFile = ModelFile {
    filename: "decoder_fp16_argmax.onnx",
    url: "", // No auto-download — must be pre-exported via export.py
};

// GLM-OCR models (manual export, no auto-download)
// Filenames match the output of py/glm-ocr/cuda/export.py.
// ONNX external data files (*.onnx.data) reference relative paths, so
// these files must stay in their export directory (use --model-cache-dir).
const GLM_VISION_ENCODER: ModelFile = ModelFile {
    filename: "vision_encoder_mha.onnx",
    url: "",
};
const GLM_EMBEDDING: ModelFile = ModelFile {
    filename: "embedding.onnx",
    url: "",
};
const GLM_LLM: ModelFile = ModelFile {
    filename: "llm.onnx",
    url: "",
};
const GLM_LLM_DECODER: ModelFile = ModelFile {
    filename: "llm_decoder_gqa.onnx",
    url: "",
};
const GLM_TOKENIZER: ModelFile = ModelFile {
    filename: "tokenizer.json",
    url: "",
};

// TableFormer V1 — split encoder/decoder/bbox ONNX (manual export, no auto-download)
const TABLEFORMER_ENCODER: ModelFile = ModelFile {
    filename: "tableformer_encoder.onnx",
    url: "", // No auto-download — must be pre-exported via py/table_former/export.py
};
const TABLEFORMER_DECODER: ModelFile = ModelFile {
    filename: "tableformer_decoder.onnx",
    url: "",
};
const TABLEFORMER_BBOX_DECODER: ModelFile = ModelFile {
    filename: "tableformer_bbox_decoder.onnx",
    url: "",
};

/// Resolved paths to all required model files.
pub struct ModelPaths {
    pub layout: PathBuf,
    pub formula_encoder: Option<PathBuf>,
    pub formula_decoder: Option<PathBuf>,
    pub formula_tokenizer: Option<PathBuf>,
    pub glm_ocr: Option<GlmOcrModelPaths>,
    pub tableformer: Option<TableFormerModelPaths>,
}

/// Resolved paths for TableFormer models.
pub struct TableFormerModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub bbox_decoder: PathBuf,
}

/// Resolved paths for GLM-OCR models (separate from pipeline models).
pub struct GlmOcrModelPaths {
    pub vision_encoder: PathBuf,
    pub embedding: PathBuf,
    pub llm: PathBuf,
    pub llm_decoder: PathBuf,
    pub tokenizer: PathBuf,
}

/// Default model cache directory.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("PAPERS_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("papers")
        .join("models")
}

/// Ensure the layout model is downloaded and return its path.
pub fn ensure_layout_model(cache_dir: &Path) -> Result<PathBuf, ExtractError> {
    std::fs::create_dir_all(cache_dir)?;
    ensure_model(cache_dir, &LAYOUT_MODEL)
}

/// Resolved paths for PP-FormulaNet models.
pub struct FormulaModelPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub tokenizer: PathBuf,
}

/// Ensure PP-FormulaNet formula models are available.
pub fn ensure_formula_models(cache_dir: &Path) -> Result<FormulaModelPaths, ExtractError> {
    std::fs::create_dir_all(cache_dir)?;
    let encoder = ensure_local_model(cache_dir, &FORMULA_ENCODER)?;
    let decoder = ensure_local_model(cache_dir, &FORMULA_DECODER)?;
    let tokenizer = ensure_model(cache_dir, &FORMULA_TOKENIZER)?;
    Ok(FormulaModelPaths { encoder, decoder, tokenizer })
}

/// Ensure all models for the selected formula/table engines are downloaded.
pub fn ensure_models(
    formula: FormulaModel,
    table: TableModel,
    cache_dir: &Path,
) -> Result<ModelPaths, ExtractError> {
    std::fs::create_dir_all(cache_dir)?;

    // Layout detection is always required
    let layout = ensure_model(cache_dir, &LAYOUT_MODEL)?;

    // Formula models (only for pp-formulanet)
    let (formula_encoder, formula_decoder, formula_tokenizer) = match formula {
        FormulaModel::PpFormulanet => {
            let enc = ensure_local_model(cache_dir, &FORMULA_ENCODER)?;
            let dec = ensure_local_model(cache_dir, &FORMULA_DECODER)?;
            let tok = ensure_model(cache_dir, &FORMULA_TOKENIZER)?;
            (Some(enc), Some(dec), Some(tok))
        }
        FormulaModel::GlmOcr => (None, None, None),
    };

    // TableFormer models
    let tableformer = if table == TableModel::TableFormer {
        Some(ensure_tableformer_models(cache_dir)?)
    } else {
        None
    };

    // GLM-OCR models (needed if either formula or table uses glm-ocr)
    let glm_ocr = if formula == FormulaModel::GlmOcr || table == TableModel::GlmOcr {
        Some(ensure_glm_ocr_models(cache_dir)?)
    } else {
        None
    };

    Ok(ModelPaths {
        layout,
        formula_encoder,
        formula_decoder,
        formula_tokenizer,
        glm_ocr,
        tableformer,
    })
}

/// Ensure a single model file exists, downloading if necessary.
fn ensure_model(cache_dir: &Path, model: &ModelFile) -> Result<PathBuf, ExtractError> {
    let local_path = cache_dir.join(model.filename);
    if local_path.exists() {
        return Ok(local_path);
    }

    tracing::info!("Downloading {} from {}...", model.filename, model.url);
    download_file(model.url, &local_path)?;
    Ok(local_path)
}

/// Ensure a local-only model file exists (no download URL available).
fn ensure_local_model(cache_dir: &Path, model: &ModelFile) -> Result<PathBuf, ExtractError> {
    let local_path = cache_dir.join(model.filename);
    if local_path.exists() {
        return Ok(local_path);
    }
    Err(ExtractError::ModelNotFound {
        path: local_path.display().to_string(),
    })
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

/// Ensure all GLM-OCR model files exist in the cache directory.
///
/// For vision encoder: tries `vision_encoder_mha.onnx` first (CUDA/MHA-fused),
/// falls back to `vision_encoder.onnx` (CPU/CoreML FP32).
/// For decoder: tries `llm_decoder_gqa.onnx` first, falls back to empty path
/// (non-CUDA backends don't need it).
pub fn ensure_glm_ocr_models(
    cache_dir: &Path,
) -> Result<GlmOcrModelPaths, ExtractError> {
    std::fs::create_dir_all(cache_dir)?;

    // Vision encoder: prefer MHA-fused, fall back to raw
    let vision_encoder = ensure_local_model(cache_dir, &GLM_VISION_ENCODER)
        .or_else(|_| {
            let raw = cache_dir.join("vision_encoder.onnx");
            if raw.exists() {
                Ok(raw)
            } else {
                Err(ExtractError::ModelNotFound {
                    path: format!(
                        "{} or {}",
                        cache_dir.join("vision_encoder_mha.onnx").display(),
                        raw.display()
                    ),
                })
            }
        })?;

    let embedding = ensure_local_model(cache_dir, &GLM_EMBEDDING)?;
    let llm = ensure_local_model(cache_dir, &GLM_LLM)?;

    // Decoder: optional (non-CUDA backends don't need it)
    let llm_decoder = ensure_local_model(cache_dir, &GLM_LLM_DECODER)
        .unwrap_or_else(|_| cache_dir.join("llm_decoder_gqa.onnx"));

    let tokenizer = ensure_local_model(cache_dir, &GLM_TOKENIZER)?;

    Ok(GlmOcrModelPaths {
        vision_encoder,
        embedding,
        llm,
        llm_decoder,
        tokenizer,
    })
}

/// Ensure all TableFormer model files exist in the cache directory.
pub fn ensure_tableformer_models(
    cache_dir: &Path,
) -> Result<TableFormerModelPaths, ExtractError> {
    let encoder = ensure_local_model(cache_dir, &TABLEFORMER_ENCODER)?;
    let decoder = ensure_local_model(cache_dir, &TABLEFORMER_DECODER)?;
    let bbox_decoder = ensure_local_model(cache_dir, &TABLEFORMER_BBOX_DECODER)?;
    Ok(TableFormerModelPaths {
        encoder,
        decoder,
        bbox_decoder,
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

/// Build a custom CUDA formula predictor from split encoder/decoder models.
pub fn build_formula_predictor(
    paths: &ModelPaths,
) -> Result<FormulaPredictor, ExtractError> {
    let enc = paths.formula_encoder.as_ref()
        .ok_or_else(|| ExtractError::Model("formula_encoder path missing".into()))?;
    let dec = paths.formula_decoder.as_ref()
        .ok_or_else(|| ExtractError::Model("formula_decoder path missing".into()))?;
    let tok = paths.formula_tokenizer.as_ref()
        .ok_or_else(|| ExtractError::Model("formula_tokenizer path missing".into()))?;
    FormulaPredictor::new(enc, dec, tok)
}

