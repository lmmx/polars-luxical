// src/registry.rs

//! Global model registry for loaded Luxical embedders.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;
use polars::prelude::{PolarsError, PolarsResult};
use pyo3::prelude::*;

use crate::embedder::{LuxicalEmbedder, LuxicalError};

/// Global registry mapping model names to loaded embedders.
static MODEL_REGISTRY: Lazy<RwLock<HashMap<String, Arc<LuxicalEmbedder>>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Default model ID (the official Luxical One model).
const DEFAULT_MODEL_ID: &str = "DatologyAI/luxical-one";

/// Known models and their download URLs.
/// Maps model_id -> (hf_url, local_filename)
fn get_model_info(model_id: &str) -> Option<(&'static str, &'static str)> {
    match model_id.to_lowercase().as_str() {
        "datologyai/luxical-one" | "luxical-one" => Some((
            "https://huggingface.co/DatologyAI/luxical-one/resolve/main/luxical_one_rc4.npz",
            "luxical_one_rc4.npz",
        )),
        _ => None,
    }
}

/// Cache directory for downloaded models.
fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("polars-luxical")
}

/// Convert a LuxicalError to a PolarsError.
fn to_polars_error(e: LuxicalError) -> PolarsError {
    PolarsError::ComputeError(e.to_string().into())
}

/// Register a model by name or HuggingFace ID.
/// If it's already loaded, this is a no-op.
#[pyfunction]
pub fn register_model(model_name: String) -> PyResult<()> {
    let mut map = MODEL_REGISTRY
        .write()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poisoned"))?;

    // Normalize the model name for lookup
    let normalized = model_name.to_lowercase();
    if map.contains_key(&normalized) {
        return Ok(());
    }

    // Try to load the model
    let embedder = load_model_impl(&model_name).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Failed to load model '{}': {}",
            model_name, e
        ))
    })?;

    map.insert(normalized, Arc::new(embedder));
    Ok(())
}

/// Clear all loaded models from the registry.
#[pyfunction]
pub fn clear_registry() -> PyResult<()> {
    let mut map = MODEL_REGISTRY
        .write()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poisoned"))?;
    map.clear();
    Ok(())
}

/// List all currently loaded model names.
#[pyfunction]
pub fn list_models() -> PyResult<Vec<String>> {
    let map = MODEL_REGISTRY
        .read()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Lock poisoned"))?;
    Ok(map.keys().cloned().collect())
}

/// Get a model from the registry, loading it if necessary.
pub fn get_or_load_model(model_name: &Option<String>) -> PolarsResult<Arc<LuxicalEmbedder>> {
    let name = model_name
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or(DEFAULT_MODEL_ID);

    let normalized = name.to_lowercase();

    // Check if already loaded
    {
        let map = MODEL_REGISTRY
            .read()
            .map_err(|_| PolarsError::ComputeError("Lock poisoned".into()))?;
        if let Some(embedder) = map.get(&normalized) {
            return Ok(embedder.clone());
        }
    }

    // Load the model
    let embedder = load_model_impl(name).map_err(to_polars_error)?;
    let arc_embedder = Arc::new(embedder);

    // Store in registry
    {
        let mut map = MODEL_REGISTRY
            .write()
            .map_err(|_| PolarsError::ComputeError("Lock poisoned".into()))?;
        map.insert(normalized, arc_embedder.clone());
    }

    Ok(arc_embedder)
}

/// Internal function to load a model from file or HuggingFace Hub.
fn load_model_impl(model_name: &str) -> Result<LuxicalEmbedder, LuxicalError> {
    // First, check if it's a local file path
    let local_path = PathBuf::from(model_name);
    if local_path.exists() {
        eprintln!("Loading model from local path: {:?}", local_path);
        return LuxicalEmbedder::load(&local_path);
    }

    // Check if it's a path with .npz extension
    let npz_path = if model_name.ends_with(".npz") {
        PathBuf::from(model_name)
    } else {
        PathBuf::from(format!("{}.npz", model_name))
    };
    if npz_path.exists() {
        eprintln!("Loading model from local path: {:?}", npz_path);
        return LuxicalEmbedder::load(&npz_path);
    }

    // Look up known model info
    let (url, filename) = get_model_info(model_name).ok_or_else(|| {
        LuxicalError::ModelNotFound(format!(
            "Unknown model '{}'. Known models: DatologyAI/luxical-one",
            model_name
        ))
    })?;

    // Try to load from cache
    let cache_path = cache_dir().join(filename);
    if cache_path.exists() {
        eprintln!("Loading model from cache: {:?}", cache_path);
        return LuxicalEmbedder::load(&cache_path);
    }

    // Download from HuggingFace Hub
    download_from_url(url, &cache_path)?;
    LuxicalEmbedder::load(&cache_path)
}

/// Download a model from a URL.
fn download_from_url(url: &str, dest_path: &PathBuf) -> Result<(), LuxicalError> {
    // Create cache directory if needed
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    eprintln!("Downloading model from: {}", url);

    let response = ureq::get(url).call().map_err(|e| {
        LuxicalError::ModelNotFound(format!("Failed to download from {}: {}", url, e))
    })?;

    let mut dest_file = std::fs::File::create(dest_path)?;
    std::io::copy(&mut response.into_reader(), &mut dest_file)?;

    eprintln!("Model downloaded to: {:?}", dest_path);
    Ok(())
}
