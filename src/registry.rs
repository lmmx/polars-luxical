// src/registry.rs
//! Global model registry for loaded Luxical embedders.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;
use polars::prelude::{PolarsError, PolarsResult};
use pyo3::prelude::*;

use crate::embedder::{LuxicalEmbedder, LuxicalError};

static MODEL_REGISTRY: Lazy<RwLock<HashMap<String, Arc<LuxicalEmbedder>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

const DEFAULT_MODEL_ID: &str = "DatologyAI/luxical-one";

/// Known models and their download info.
fn get_model_info(model_id: &str) -> Option<ModelInfo> {
    match model_id.to_lowercase().as_str() {
        "datologyai/luxical-one" | "luxical-one" => Some(ModelInfo {
            // Updated paths to include artifacts/luxical-one subdirectory
            safetensors_url: "https://huggingface.co/DatologyAI/luxical-one/resolve/main/artifacts/luxical-one/model.safetensors",
            safetensors_filename: "model.safetensors",
            npz_url: "https://huggingface.co/DatologyAI/luxical-one/resolve/main/luxical_one_rc4.npz",
            npz_filename: "luxical_one_rc4.npz",
        }),
        _ => None,
    }
}

struct ModelInfo {
    npz_url: &'static str,
    npz_filename: &'static str,
    safetensors_url: &'static str,
    safetensors_filename: &'static str,
}

fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("polars-luxical")
}

fn to_polars_error(e: LuxicalError) -> PolarsError {
    PolarsError::ComputeError(e.to_string().into())
}

#[pyfunction]
pub fn register_model(model_name: String) -> PyResult<()> {
    let mut map = MODEL_REGISTRY
        .write()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poisoned"))?;

    let normalized = model_name.to_lowercase();
    if map.contains_key(&normalized) {
        return Ok(());
    }

    let embedder = load_model_impl(&model_name).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load model '{}': {}",
            model_name, e
        ))
    })?;

    map.insert(normalized, Arc::new(embedder));
    Ok(())
}

#[pyfunction]
pub fn clear_registry() -> PyResult<()> {
    let mut map = MODEL_REGISTRY
        .write()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poisoned"))?;
    map.clear();
    Ok(())
}

#[pyfunction]
pub fn list_models() -> PyResult<Vec<String>> {
    let map = MODEL_REGISTRY
        .read()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poisoned"))?;
    Ok(map.keys().cloned().collect())
}

pub fn get_or_load_model(model_name: &Option<String>) -> PolarsResult<Arc<LuxicalEmbedder>> {
    let name = model_name
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or(DEFAULT_MODEL_ID);

    let normalized = name.to_lowercase();

    {
        let map = MODEL_REGISTRY
            .read()
            .map_err(|_| PolarsError::ComputeError("Lock poisoned".into()))?;
        if let Some(embedder) = map.get(&normalized) {
            return Ok(embedder.clone());
        }
    }

    let embedder = load_model_impl(name).map_err(to_polars_error)?;
    let arc_embedder = Arc::new(embedder);

    {
        let mut map = MODEL_REGISTRY
            .write()
            .map_err(|_| PolarsError::ComputeError("Lock poisoned".into()))?;
        map.insert(normalized, arc_embedder.clone());
    }

    Ok(arc_embedder)
}

fn load_model_impl(model_name: &str) -> Result<LuxicalEmbedder, LuxicalError> {
    // Check local paths first
    let local_path = PathBuf::from(model_name);
    if local_path.exists() {
        eprintln!("Loading model from local path: {:?}", local_path);
        return LuxicalEmbedder::load(&local_path);
    }

    // Look up known model info
    let info = get_model_info(model_name).ok_or_else(|| {
        LuxicalError::ModelNotFound(format!(
            "Unknown model '{}'. Known models: DatologyAI/luxical-one",
            model_name
        ))
    })?;

    let cache = cache_dir();

    // Prefer safetensors (faster loading via mmap)
    let safetensors_path = cache.join(info.safetensors_filename);
    if safetensors_path.exists() {
        eprintln!("Loading model from cache (safetensors): {:?}", safetensors_path);
        return LuxicalEmbedder::load(&safetensors_path);
    }

    // Fall back to NPZ
    let npz_path = cache.join(info.npz_filename);
    if npz_path.exists() {
        eprintln!("Loading model from cache (npz): {:?}", npz_path);
        return LuxicalEmbedder::load(&npz_path);
    }

    // Download - try safetensors first (preferred format)
    eprintln!("Attempting to download safetensors format...");
    match download_from_url(info.safetensors_url, &safetensors_path) {
        Ok(()) => {
            return LuxicalEmbedder::load(&safetensors_path);
        }
        Err(e) => {
            eprintln!("Safetensors download failed: {}", e);
            eprintln!("Falling back to NPZ format...");
        }
    }

    // Fall back to NPZ download
    download_from_url(info.npz_url, &npz_path)?;
    LuxicalEmbedder::load(&npz_path)
}

fn download_from_url(url: &str, dest_path: &PathBuf) -> Result<(), LuxicalError> {
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    eprintln!("Downloading model from: {}", url);

    let response = ureq::get(url).call().map_err(|e| {
        LuxicalError::ModelNotFound(format!("Failed to download from {}: {}", url, e))
    })?;

    // ureq v3: get the body, then call into_reader() to get something that implements Read
    let (_, body) = response.into_parts();
    let mut reader = body
        .into_with_config()
        .limit(1024 * 1024 * 1024) // 1GB limit for model files
        .reader();

    let mut dest_file = std::fs::File::create(dest_path)?;
    std::io::copy(&mut reader, &mut dest_file)?;

    eprintln!("Model downloaded to: {:?}", dest_path);
    Ok(())
}
