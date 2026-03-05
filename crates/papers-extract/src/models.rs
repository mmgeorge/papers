use std::path::{Path, PathBuf};

use oar_ocr::core::config::{OrtExecutionProvider, OrtGraphOptimizationLevel, OrtSessionConfig};
use oar_ocr::predictors::TableStructureRecognitionPredictor;

use crate::error::ExtractError;
use crate::formula::FormulaPredictor;
use crate::glm_ocr::{GlmOcrConfig, GlmOcrPredictor};
use crate::Quality;

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

// Table structure (Fast mode — wireless/general)
const SLANET_PLUS: ModelFile = ModelFile {
    filename: "slanet_plus.onnx",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanet_plus.onnx",
};

// Table structure dictionary (always required with table models)
const TABLE_DICT: ModelFile = ModelFile {
    filename: "table_structure_dict_ch.txt",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/table_structure_dict_ch.txt",
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

// Table classification — wired vs wireless (Quality mode only)
const TABLE_CLASSIFIER: ModelFile = ModelFile {
    filename: "pp-lcnet_x1_0_table_cls.onnx",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-lcnet_x1_0_table_cls.onnx",
};

// Table structure — wired (Quality mode only)
const SLANEXT_WIRED: ModelFile = ModelFile {
    filename: "slanext_wired.onnx",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/slanext_wired.onnx",
};

/// Resolved paths to all required model files.
pub struct ModelPaths {
    pub layout: PathBuf,
    pub slanet_plus: PathBuf,
    pub table_dict: PathBuf,
    pub formula_encoder: PathBuf,
    pub formula_decoder: PathBuf,
    pub formula_tokenizer: PathBuf,
    // Quality-mode extras
    pub table_classifier: Option<PathBuf>,
    pub slanext_wired: Option<PathBuf>,
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

/// Ensure all models for the given quality mode are downloaded and return their paths.
pub fn ensure_models(
    quality: Quality,
    cache_dir: &Path,
) -> Result<ModelPaths, ExtractError> {
    std::fs::create_dir_all(cache_dir)?;

    // Always required
    let layout = ensure_model(cache_dir, &LAYOUT_MODEL)?;
    let slanet_plus = ensure_model(cache_dir, &SLANET_PLUS)?;
    let table_dict = ensure_model(cache_dir, &TABLE_DICT)?;
    let formula_tokenizer = ensure_model(cache_dir, &FORMULA_TOKENIZER)?;

    // Formula encoder/decoder (no auto-download, just check existence)
    let formula_encoder = ensure_local_model(cache_dir, &FORMULA_ENCODER)?;
    let formula_decoder = ensure_local_model(cache_dir, &FORMULA_DECODER)?;

    // Table models selected by --quality
    let (table_classifier, slanext_wired) = match quality {
        Quality::Fast => (None, None),
        Quality::Quality => {
            let classifier = ensure_model(cache_dir, &TABLE_CLASSIFIER)?;
            let wired = ensure_model(cache_dir, &SLANEXT_WIRED)?;
            (Some(classifier), Some(wired))
        }
    };

    Ok(ModelPaths {
        layout,
        slanet_plus,
        table_dict,
        formula_encoder,
        formula_decoder,
        formula_tokenizer,
        table_classifier,
        slanext_wired,
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

/// Initialize ORT with a custom runtime library if available.
///
/// Looks for `ORT_DYLIB_PATH` env var first, then checks for a local
/// `onnxruntime/lib/onnxruntime.dll` (Windows) or `libonnxruntime.so` (Linux).
/// Must be called before any ORT sessions are created.
pub fn init_ort_runtime() -> Result<(), ExtractError> {
    use std::sync::Once;
    static INIT: Once = Once::new();
    let mut init_err: Option<String> = None;

    INIT.call_once(|| {
        let dylib_path = std::env::var("ORT_DYLIB_PATH").ok().or_else(|| {
            #[cfg(target_os = "windows")]
            let candidate = Path::new("onnxruntime/lib/onnxruntime.dll");
            #[cfg(not(target_os = "windows"))]
            let candidate = Path::new("onnxruntime/lib/libonnxruntime.so");

            if candidate.exists() {
                Some(candidate.to_string_lossy().into_owned())
            } else {
                None
            }
        });

        if let Some(path) = dylib_path {
            eprintln!("Loading ORT runtime from: {path}");
            match ort::init_from(&path) {
                Ok(builder) => {
                    builder.commit();
                }
                Err(e) => {
                    init_err = Some(format!("ORT init_from failed for '{path}': {e}"));
                }
            }
        }
    });

    if let Some(err) = init_err {
        Err(ExtractError::Model(err))
    } else {
        Ok(())
    }
}

/// Build the platform-specific ORT session configuration (for oar-ocr predictors).
pub fn ort_config() -> OrtSessionConfig {
    let providers = platform_execution_providers();
    OrtSessionConfig::new()
        .with_execution_providers(providers)
        .with_optimization_level(OrtGraphOptimizationLevel::All)
}

/// Get the execution providers for the current platform (for oar-ocr).
fn platform_execution_providers() -> Vec<OrtExecutionProvider> {
    let mut providers = Vec::new();

    #[cfg(target_os = "windows")]
    {
        providers.push(OrtExecutionProvider::CUDA {
            device_id: None,
            gpu_mem_limit: None,
            arena_extend_strategy: None,
            cudnn_conv_algo_search: None,
            cudnn_conv_use_max_workspace: None,
        });
        providers.push(OrtExecutionProvider::DirectML { device_id: None });
    }

    #[cfg(target_os = "macos")]
    providers.push(OrtExecutionProvider::CoreML {
        ane_only: None,
        subgraphs: None,
    });

    providers.push(OrtExecutionProvider::CPU);
    providers
}

/// Build a custom CUDA formula predictor from split encoder/decoder models.
pub fn build_formula_predictor(
    paths: &ModelPaths,
) -> Result<FormulaPredictor, ExtractError> {
    FormulaPredictor::new(
        &paths.formula_encoder,
        &paths.formula_decoder,
        &paths.formula_tokenizer,
    )
}

/// Build a standalone table structure recognition predictor.
pub fn build_table_predictor(
    paths: &ModelPaths,
) -> Result<TableStructureRecognitionPredictor, ExtractError> {
    let config = ort_config();
    TableStructureRecognitionPredictor::builder()
        .dict_path(&paths.table_dict)
        .with_ort_config(config)
        .build(&paths.slanet_plus)
        .map_err(|e| ExtractError::Model(format!("Failed to build table predictor: {e}")))
}
