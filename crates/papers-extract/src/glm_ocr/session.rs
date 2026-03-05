//! DRY session builder helpers for GLM-OCR ONNX sessions.

use std::path::Path;

use ort::execution_providers::ExecutionProviderDispatch;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::error::ExtractError;

/// Build an ORT session with the given optimization level and execution provider.
pub(crate) fn build_session(
    path: &Path,
    opt_level: GraphOptimizationLevel,
    ep: impl Into<ExecutionProviderDispatch>,
) -> Result<Session, ExtractError> {
    Session::builder()
        .map_err(|e| ExtractError::Model(format!("session builder: {e}")))?
        .with_optimization_level(opt_level)
        .map_err(|e| ExtractError::Model(format!("opt level: {e}")))?
        .with_execution_providers([ep.into()])
        .map_err(|e| ExtractError::Model(format!("EP: {e}")))?
        .commit_from_file(path)
        .map_err(|e| ExtractError::Model(format!("load {}: {e}", path.display())))
}
