// src/embedder/model.rs
//! Core model struct and error types.

use std::collections::HashMap;
use std::path::Path;

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;
use thiserror::Error;
use tokenizers::Tokenizer;

use super::npz_loader::load_from_npz;
use super::safetensors_loader::load_from_safetensors;
use crate::ngrams::BuildIdentityHasher;

/// Map from n-gram hash to vocabulary index. The keys are FNV hashes, so no rehashing.
pub type NgramIndex = HashMap<i64, u32, BuildIdentityHasher>;

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
    pub(crate) ngram_hash_to_idx: NgramIndex,
    /// IDF values for each vocabulary term
    pub(crate) idf_values: Array1<f32>,
    /// First (sparse) layer stored transposed as vocab_size x hidden_dim, so the
    /// weights of one n-gram are a contiguous row
    pub(crate) first_layer_t: Array2<f32>,
    /// Remaining dense layers (each is output_dim x input_dim)
    pub(crate) dense_layers: Vec<Array2<f32>>,
    /// Output embedding dimension
    pub(crate) output_dim: usize,
}

impl LuxicalEmbedder {
    /// Assemble an embedder from loaded parts.
    /// `first_layer` is hidden_dim x vocab_size, as stored in the model files.
    pub(crate) fn new(
        tokenizer: Tokenizer,
        max_ngram_length: usize,
        ngram_hash_to_idx: NgramIndex,
        idf_values: Array1<f32>,
        first_layer: ArrayView2<f32>,
        dense_layers: Vec<Array2<f32>>,
    ) -> Result<Self> {
        let output_dim = dense_layers
            .last()
            .map(|l| l.nrows())
            .unwrap_or(first_layer.nrows());
        Ok(Self {
            tokenizer,
            max_ngram_length,
            ngram_hash_to_idx,
            idf_values,
            first_layer_t: transpose(first_layer),
            dense_layers,
            output_dim,
        })
    }

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

/// Transpose a wide row-major matrix in cache-sized column blocks, in parallel.
fn transpose(m: ArrayView2<f32>) -> Array2<f32> {
    const BLOCK: usize = 1024;
    let (rows, cols) = m.dim();
    let mut out = vec![0f32; rows * cols];
    out.par_chunks_mut(BLOCK * rows)
        .enumerate()
        .for_each(|(block, chunk)| {
            let col0 = block * BLOCK;
            let width = chunk.len() / rows;
            for r in 0..rows {
                let src = m.row(r);
                for c in 0..width {
                    chunk[c * rows + r] = src[col0 + c];
                }
            }
        });
    Array2::from_shape_vec((cols, rows), out).expect("transpose shape")
}
