use std::io;
use std::path::PathBuf;

pub const VALID_MODELS: &[&str] = &["nomic-embed-text-v2-moe"];

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unknown model: {0}")]
    UnknownModel(String),
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct PapersConfig {
    pub embedding_model: String,
}

impl Default for PapersConfig {
    fn default() -> Self {
        Self {
            embedding_model: "nomic-embed-text-v2-moe".to_string(),
        }
    }
}

impl PapersConfig {
    /// Returns `<config_dir>/.papers/config.json`.
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".papers")
            .join("config.json")
    }

    /// Loads from disk. Returns `Default` if file missing. Errors on bad JSON or I/O failure.
    pub fn load() -> Result<Self, ConfigError> {
        let path = Self::config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(&path)?;
        let cfg = serde_json::from_slice(&bytes)?;
        Ok(cfg)
    }

    /// Writes to disk, creating parent directories as needed.
    pub fn save(&self) -> Result<(), ConfigError> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_vec_pretty(self)?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Returns `Err(ConfigError::UnknownModel)` if `name` is not in `VALID_MODELS`.
    pub fn validate_model(name: &str) -> Result<(), ConfigError> {
        if VALID_MODELS.contains(&name) {
            Ok(())
        } else {
            Err(ConfigError::UnknownModel(name.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_config_default() {
        let cfg = PapersConfig::default();
        assert_eq!(cfg.embedding_model, "nomic-embed-text-v2-moe");
    }

    #[test]
    fn test_config_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        let cfg = PapersConfig {
            embedding_model: "nomic-embed-text-v2-moe".to_string(),
        };
        let json = serde_json::to_vec_pretty(&cfg).unwrap();
        std::fs::write(&path, &json).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let loaded: PapersConfig = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(loaded.embedding_model, cfg.embedding_model);
    }

    #[test]
    fn test_config_missing_file_returns_default() {
        // Test that a missing path leads to the default being used (simulating load behavior).
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.json");
        assert!(!path.exists());
        // Simulate load logic: if path doesn't exist, return default.
        let cfg = if path.exists() {
            let bytes = std::fs::read(&path).unwrap();
            serde_json::from_slice::<PapersConfig>(&bytes).unwrap()
        } else {
            PapersConfig::default()
        };
        assert_eq!(cfg.embedding_model, "nomic-embed-text-v2-moe");
    }

    #[test]
    fn test_config_invalid_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, b"not valid json{{{").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let result: Result<PapersConfig, _> = serde_json::from_slice(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_model_accepted() {
        assert!(PapersConfig::validate_model("nomic-embed-text-v2-moe").is_ok());
    }

    #[test]
    fn test_invalid_model_rejected() {
        let err = PapersConfig::validate_model("gpt-4").unwrap_err();
        assert!(matches!(err, ConfigError::UnknownModel(ref s) if s == "gpt-4"));
    }

    #[test]
    fn test_config_path_is_platform_appropriate() {
        let path = PapersConfig::config_path();
        let s = path.to_string_lossy();
        assert!(s.contains(".papers"));
        assert!(s.ends_with("config.json"));
    }
}
