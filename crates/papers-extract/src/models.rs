use std::path::{Path, PathBuf};

use oar_ocr::core::config::OrtExecutionProvider;
use oar_ocr::core::config::OrtSessionConfig;
use oar_ocr::predictors::{FormulaRecognitionPredictor, TableStructureRecognitionPredictor};

use crate::error::ExtractError;
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

// Formula recognition — small (Fast mode)
const FORMULANET_S: ModelFile = ModelFile {
    filename: "pp-formulanet_plus-s.onnx",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-s.onnx",
};

// Formula tokenizer (required for all formula models)
const FORMULA_TOKENIZER: ModelFile = ModelFile {
    filename: "unimernet_tokenizer.json",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/unimernet_tokenizer.json",
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

// Formula recognition — large (Quality mode only)
const FORMULANET_L: ModelFile = ModelFile {
    filename: "pp-formulanet_plus-l.onnx",
    url: "https://github.com/GreatV/oar-ocr/releases/download/v0.3.0/pp-formulanet_plus-l.onnx",
};

/// Resolved paths to all required model files.
pub struct ModelPaths {
    pub layout: PathBuf,
    pub slanet_plus: PathBuf,
    pub table_dict: PathBuf,
    pub formula: PathBuf,
    pub formula_tokenizer: PathBuf,
    // Quality-mode extras
    pub table_classifier: Option<PathBuf>,
    pub slanext_wired: Option<PathBuf>,
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

    let (formula, table_classifier, slanext_wired) = match quality {
        Quality::Fast => {
            let formula = ensure_model(cache_dir, &FORMULANET_S)?;
            (formula, None, None)
        }
        Quality::Quality => {
            let formula = ensure_model(cache_dir, &FORMULANET_L)?;
            let classifier = ensure_model(cache_dir, &TABLE_CLASSIFIER)?;
            let wired = ensure_model(cache_dir, &SLANEXT_WIRED)?;
            (formula, Some(classifier), Some(wired))
        }
    };

    Ok(ModelPaths {
        layout,
        slanet_plus,
        table_dict,
        formula,
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

/// Build a direct layout detector from the pp-doclayoutv3.onnx model.
pub fn build_layout_detector(
    model_path: &Path,
) -> Result<crate::layout::LayoutDetector, ExtractError> {
    crate::layout::LayoutDetector::new(model_path)
}

/// Build the platform-specific ORT session configuration.
pub fn ort_config() -> OrtSessionConfig {
    let providers = platform_execution_providers();
    OrtSessionConfig::new().with_execution_providers(providers)
}

/// Get the execution providers for the current platform.
fn platform_execution_providers() -> Vec<OrtExecutionProvider> {
    let mut providers = Vec::new();

    #[cfg(target_os = "windows")]
    providers.push(OrtExecutionProvider::DirectML { device_id: None });

    #[cfg(target_os = "macos")]
    providers.push(OrtExecutionProvider::CoreML {
        ane_only: None,
        subgraphs: None,
    });

    providers.push(OrtExecutionProvider::CPU);
    providers
}

/// Build a standalone formula recognition predictor.
pub fn build_formula_predictor(
    paths: &ModelPaths,
    quality: Quality,
) -> Result<FormulaRecognitionPredictor, ExtractError> {
    let config = ort_config();
    let name = match quality {
        Quality::Fast => "PP-FormulaNet_plus-S",
        Quality::Quality => "PP-FormulaNet_plus-L",
    };
    FormulaRecognitionPredictor::builder()
        .model_name(name)
        .tokenizer_path(&paths.formula_tokenizer)
        .with_ort_config(config)
        .build(&paths.formula)
        .map_err(|e| ExtractError::Model(format!("Failed to build formula predictor: {e}")))
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
