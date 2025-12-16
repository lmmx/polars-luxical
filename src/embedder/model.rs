// src/embedder/model.rs
//! Core model struct and error types.

use std::collections::HashMap;
use std::path::Path;

use ndarray::{Array1, Array2};
use thiserror::Error;
use tokenizers::Tokenizer;

use super::npz_loader::load_from_npz;
use super::safetensors_loader::load_from_safetensors;

#[derive(Error, Debug)]
pub enum LuxicalError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Safetensors error: {0}")]
    Safetensors(String),
}

pub type Result<T> = std::result::Result<T, LuxicalError>;

/// A loaded Luxical embedding model.
pub struct LuxicalEmbedder {
    /// The tokenizer (from HuggingFace tokenizers)
    pub(crate) tokenizer: Tokenizer,
    /// Maximum n-gram length to extract
    pub(crate) max_ngram_length: usize,
    /// Map from n-gram hash to vocabulary index
    pub(crate) ngram_hash_to_idx: HashMap<i64, u32>,
    /// IDF values for each vocabulary term
    pub(crate) idf_values: Array1<f32>,
    /// Neural network layers (each is output_dim x input_dim)
    pub(crate) layers: Vec<Array2<f32>>,
    /// Output embedding dimension
    pub(crate) output_dim: usize,
}

impl LuxicalEmbedder {
    /// Load a Luxical model from a file path.
    /// Automatically detects format based on file extension or directory structure.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Check if it's a safetensors file or directory containing one
        if path.extension().map(|e| e == "safetensors").unwrap_or(false) {
            return load_from_safetensors(path);
        }

        // Check if it's a directory with model.safetensors
        if path.is_dir() {
            let safetensors_path = path.join("model.safetensors");
            if safetensors_path.exists() {
                return load_from_safetensors(&safetensors_path);
            }
        }

        // Check for NPZ file
        if path.extension().map(|e| e == "npz").unwrap_or(false) {
            return load_from_npz(path);
        }

        // Try NPZ as fallback
        if path.exists() {
            return load_from_npz(path);
        }

        Err(LuxicalError::ModelNotFound(format!(
            "No valid model found at: {}",
            path.display()
        )))
    }

    /// Get the output embedding dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.ngram_hash_to_idx.len()
    }

    /// Get max ngram length.
    pub fn max_ngram_length(&self) -> usize {
        self.max_ngram_length
    }
}
