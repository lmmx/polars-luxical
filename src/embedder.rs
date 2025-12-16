//! Core Luxical embedder implementation in pure Rust.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use ndarray::{Array1, Array2, Axis};
use sprs::{CsMat, TriMat};
use thiserror::Error;
use tokenizers::Tokenizer;
use bytemuck::cast_slice;

use crate::ngrams::extract_ngrams_hashed;
use crate::tfidf::apply_tfidf_and_normalize;

#[derive(Error, Debug)]
pub enum LuxicalError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("NPZ parse error: {0}")]
    NpzParse(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

pub type Result<T> = std::result::Result<T, LuxicalError>;

/// A loaded Luxical embedding model.
pub struct LuxicalEmbedder {
    /// The tokenizer (from HuggingFace tokenizers)
    tokenizer: Tokenizer,
    /// Maximum n-gram length to extract
    max_ngram_length: usize,
    /// Map from n-gram hash to vocabulary index
    ngram_hash_to_idx: HashMap<i64, u32>,
    /// IDF values for each vocabulary term
    idf_values: Array1<f32>,
    /// Neural network layers (each is output_dim x input_dim)
    layers: Vec<Array2<f32>>,
    /// Output embedding dimension
    output_dim: usize,
}

impl LuxicalEmbedder {
    /// Load a Luxical model from an NPZ file path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::load_from_reader(reader)
    }

    /// Load from a reader (for flexibility with different sources).
    fn load_from_reader<R: Read + std::io::Seek>(reader: R) -> Result<Self> {
        let mut npz = NpzReader::new(reader)?;

        // Load tokenizer from JSON bytes (stored as uint8 array)
        let tokenizer_bytes: Vec<u8> = npz.read_array_u8("tokenizer")?;
        let tokenizer_json = String::from_utf8(tokenizer_bytes)
            .map_err(|e| LuxicalError::NpzParse(format!("Invalid UTF-8 in tokenizer: {}", e)))?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .map_err(|e| LuxicalError::Tokenizer(e.to_string()))?;

        // Load recognized ngrams to determine max_ngram_length
        // Note: stored as uint32 in the file, shape (2000000, 5)
        let (recognized_ngrams_flat, recognized_ngrams_shape) = npz.read_array_u32("recognized_ngrams")?;
        let max_ngram_length = if recognized_ngrams_shape.len() == 2 {
            recognized_ngrams_shape[1]
        } else {
            5 // default
        };

        // Build ngram hash to index map
        let (ngram_hash_to_idx_keys, _) = npz.read_array_i64("ngram_hash_to_ngram_idx_keys")?;
        let (ngram_hash_to_idx_values, _) = npz.read_array_u32("ngram_hash_to_ngram_idx_values")?;

        let mut ngram_hash_to_idx = HashMap::with_capacity(ngram_hash_to_idx_keys.len());
        for (k, v) in ngram_hash_to_idx_keys
            .iter()
            .zip(ngram_hash_to_idx_values.iter())
        {
            ngram_hash_to_idx.insert(*k, *v);
        }

        // Load IDF values
        let (idf_values_vec, _) = npz.read_array_f32("idf_values")?;
        let idf_values = Array1::from_vec(idf_values_vec);

        // Load neural network layers
        // num_nn_layers is stored as shape (1,), not a scalar
        let (num_layers_vec, _) = npz.read_array_i64("num_nn_layers")?;
        let num_layers = num_layers_vec[0] as usize;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let (layer_vec, layer_shape) = npz.read_array_f32(&format!("nn_layer_{}", i))?;
            if layer_shape.len() != 2 {
                return Err(LuxicalError::NpzParse(format!(
                    "Expected 2D layer, got {}D",
                    layer_shape.len()
                )));
            }
            let layer = Array2::from_shape_vec((layer_shape[0], layer_shape[1]), layer_vec)
                .map_err(|e| LuxicalError::NpzParse(e.to_string()))?;
            layers.push(layer);
        }

        let output_dim = layers.last().map(|l| l.nrows()).unwrap_or(0);

        Ok(Self {
            tokenizer,
            max_ngram_length,
            ngram_hash_to_idx,
            idf_values,
            layers,
            output_dim,
        })
    }

    /// Get the output embedding dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.ngram_hash_to_idx.len()
    }

    /// Embed a batch of texts, returning a 2D array of shape (n_texts, output_dim).
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Array2<f32>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, self.output_dim)));
        }

        // 1. Tokenize all texts
        let token_ids: Vec<Vec<u32>> = texts
            .iter()
            .map(|text| {
                let encoding = self
                    .tokenizer
                    .encode(*text, false)
                    .map_err(|e| LuxicalError::Tokenizer(e.to_string()))?;
                Ok(encoding.get_ids().to_vec())
            })
            .collect::<Result<Vec<_>>>()?;

        // 2. Build sparse BoW matrix with n-gram hashing
        let bow = self.build_bow_matrix(&token_ids);

        // 3. Apply TF-IDF weighting
        let tfidf = apply_tfidf_and_normalize(&bow, self.idf_values.view());

        // 4. Forward pass through MLP
        let embeddings = self.forward(&tfidf)?;

        Ok(embeddings)
    }

    /// Build a sparse BoW (bag-of-words) matrix from tokenized documents.
    fn build_bow_matrix(&self, token_ids: &[Vec<u32>]) -> CsMat<f32> {
        let n_docs = token_ids.len();
        let vocab_size = self.vocab_size();

        // Use a triplet matrix for efficient construction
        let mut triplets = TriMat::new((n_docs, vocab_size));

        for (doc_idx, tokens) in token_ids.iter().enumerate() {
            // Count n-grams for this document
            let mut counts: HashMap<u32, u32> = HashMap::new();

            for ngram_hash in extract_ngrams_hashed(tokens, self.max_ngram_length) {
                if let Some(&vocab_idx) = self.ngram_hash_to_idx.get(&ngram_hash) {
                    *counts.entry(vocab_idx).or_insert(0) += 1;
                }
            }

            // Add to triplet matrix
            for (vocab_idx, count) in counts {
                triplets.add_triplet(doc_idx, vocab_idx as usize, count as f32);
            }
        }

        triplets.to_csr()
    }

    /// Forward pass through the MLP layers.
    fn forward(&self, tfidf: &CsMat<f32>) -> Result<Array2<f32>> {
        if self.layers.is_empty() {
            return Err(LuxicalError::NpzParse("No layers in model".to_string()));
        }

        // First layer: sparse @ dense.T
        let mut hidden = sparse_dense_matmul(tfidf, &self.layers[0]);
        relu_inplace(&mut hidden);
        normalize_rows_inplace(&mut hidden);

        // Hidden layers
        for layer in &self.layers[1..self.layers.len() - 1] {
            hidden = dense_matmul(&hidden, layer);
            relu_inplace(&mut hidden);
            normalize_rows_inplace(&mut hidden);
        }

        // Final layer (no ReLU)
        if self.layers.len() > 1 {
            hidden = dense_matmul(&hidden, self.layers.last().unwrap());
            normalize_rows_inplace(&mut hidden);
        }

        Ok(hidden)
    }
}

/// Sparse matrix (CSR) times dense matrix (stored as output_dim x input_dim).
/// Returns dense matrix of shape (n_rows, output_dim).
fn sparse_dense_matmul(sparse: &CsMat<f32>, dense: &Array2<f32>) -> Array2<f32> {
    let n_rows = sparse.rows();
    let output_dim = dense.nrows();

    let mut result = Array2::zeros((n_rows, output_dim));

    // For each row in the sparse matrix
    for (row_idx, row_vec) in sparse.outer_iterator().enumerate() {
        // For each non-zero element in this row
        for (col_idx, &val) in row_vec.iter() {
            // Add val * dense[col_idx, :] to result[row_idx, :]
            let dense_col = dense.column(col_idx);
            for (out_idx, &dense_val) in dense_col.iter().enumerate() {
                result[[row_idx, out_idx]] += val * dense_val;
            }
        }
    }

    result
}

/// Dense matrix multiplication: A @ B.T where B is stored as (output_dim, input_dim).
fn dense_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    a.dot(&b.t())
}

/// Apply ReLU in-place.
fn relu_inplace(arr: &mut Array2<f32>) {
    arr.mapv_inplace(|x| x.max(0.0));
}

/// L2 normalize each row in-place.
fn normalize_rows_inplace(arr: &mut Array2<f32>) {
    for mut row in arr.axis_iter_mut(Axis(0)) {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|x| x / norm);
        }
    }
}

// ============================================================================
// NPZ Reader - optimized implementation for loading Luxical model files
// ============================================================================

use std::io::Seek;

struct NpzReader<R: Read + Seek> {
    archive: zip::ZipArchive<R>,
}

impl<R: Read + Seek> NpzReader<R> {
    fn new(reader: R) -> Result<Self> {
        let archive =
            zip::ZipArchive::new(reader).map_err(|e| LuxicalError::NpzParse(e.to_string()))?;
        Ok(Self { archive })
    }

    fn read_npy_header(data: &[u8]) -> Result<(Vec<usize>, String, usize)> {
        // NPY format: magic + version + header_len + header
        if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
            return Err(LuxicalError::NpzParse("Invalid NPY magic".to_string()));
        }

        let version_major = data[6];
        let header_len = if version_major == 1 {
            u16::from_le_bytes([data[8], data[9]]) as usize
        } else {
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
        };

        let header_start = if version_major == 1 { 10 } else { 12 };
        let header_end = header_start + header_len;
        let header_str = std::str::from_utf8(&data[header_start..header_end])
            .map_err(|e| LuxicalError::NpzParse(format!("Invalid header UTF-8: {}", e)))?;

        // Parse the header dict
        let shape = Self::parse_shape(header_str)?;
        let dtype = Self::parse_dtype(header_str)?;

        Ok((shape, dtype, header_end))
    }

    fn parse_shape(header: &str) -> Result<Vec<usize>> {
        let shape_marker = "'shape':";
        let shape_start = header
            .find(shape_marker)
            .ok_or_else(|| LuxicalError::NpzParse("No shape in header".to_string()))?;
        let rest = &header[shape_start + shape_marker.len()..];
        let rest = rest.trim_start();

        let paren_start = rest
            .find('(')
            .ok_or_else(|| LuxicalError::NpzParse("No shape tuple".to_string()))?;
        let paren_end = rest
            .find(')')
            .ok_or_else(|| LuxicalError::NpzParse("No shape tuple end".to_string()))?;
        let shape_str = &rest[paren_start + 1..paren_end];

        let shape: Vec<usize> = shape_str
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                if s.is_empty() {
                    None
                } else {
                    s.parse().ok()
                }
            })
            .collect();

        Ok(shape)
    }

    fn parse_dtype(header: &str) -> Result<String> {
        let descr_marker = "'descr':";
        let descr_start = header
            .find(descr_marker)
            .ok_or_else(|| LuxicalError::NpzParse("No descr in header".to_string()))?;
        let rest = &header[descr_start + descr_marker.len()..];
        let rest = rest.trim_start();

        if !rest.starts_with('\'') {
            return Err(LuxicalError::NpzParse(format!(
                "Expected quote after 'descr':', got: {}",
                &rest[..rest.len().min(20)]
            )));
        }
        let rest = &rest[1..];

        let quote_end = rest
            .find('\'')
            .ok_or_else(|| LuxicalError::NpzParse("No descr end quote".to_string()))?;
        Ok(rest[..quote_end].to_string())
    }

    /// Read raw bytes for an array from the NPZ
    fn read_raw_array(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, String)> {
        let npy_name = format!("{}.npy", name);
        let mut file = self
            .archive
            .by_name(&npy_name)
            .map_err(|_| LuxicalError::NpzParse(format!("Array '{}' not found", name)))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let (shape, dtype, data_start) = Self::read_npy_header(&data)?;

        // Return just the data portion
        Ok((data[data_start..].to_vec(), shape, dtype))
    }

    /// Read a u8 array (for tokenizer bytes)
    fn read_array_u8(&mut self, name: &str) -> Result<Vec<u8>> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("u1") && !dtype.contains("uint8") {
            return Err(LuxicalError::NpzParse(format!(
                "Expected uint8, got {}",
                dtype
            )));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        if data.len() < n_elements {
            return Err(LuxicalError::NpzParse("Data too short for u8 array".to_string()));
        }

        Ok(data[..n_elements].to_vec())
    }

    /// Read a f32 array - FAST bulk read
    fn read_array_f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("f4") && !dtype.contains("float32") {
            return Err(LuxicalError::NpzParse(format!(
                "Expected float32, got {}",
                dtype
            )));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = n_elements * 4;

        if data.len() < expected_bytes {
            return Err(LuxicalError::NpzParse(format!(
                "Data too short: {} < {}",
                data.len(),
                expected_bytes
            )));
        }

        // Zero-copy cast (requires data alignment, which NPY provides)
        let floats: &[f32] = cast_slice(&data[..expected_bytes]);
        Ok((floats.to_vec(), shape))
    }

    /// Read an i64 array - FAST bulk read
    fn read_array_i64(&mut self, name: &str) -> Result<(Vec<i64>, Vec<usize>)> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("i8") && !dtype.contains("int64") {
            return Err(LuxicalError::NpzParse(format!(
                "Expected int64, got {}",
                dtype
            )));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = n_elements * 8;

        if data.len() < expected_bytes {
            return Err(LuxicalError::NpzParse(format!(
                "Data too short: {} < {}",
                data.len(),
                expected_bytes
            )));
        }

        // FAST: reinterpret bytes as i64
        let result: Vec<i64> = data[..expected_bytes]
            .chunks_exact(8)
            .map(|chunk| {
                i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ])
            })
            .collect();

        Ok((result, shape))
    }

    /// Read a u32 array - FAST bulk read
    fn read_array_u32(&mut self, name: &str) -> Result<(Vec<u32>, Vec<usize>)> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("u4") && !dtype.contains("uint32") {
            return Err(LuxicalError::NpzParse(format!(
                "Expected uint32, got {}",
                dtype
            )));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = n_elements * 4;

        if data.len() < expected_bytes {
            return Err(LuxicalError::NpzParse(format!(
                "Data too short: {} < {}",
                data.len(),
                expected_bytes
            )));
        }

        // FAST: reinterpret bytes as u32
        let result: Vec<u32> = data[..expected_bytes]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok((result, shape))
    }
}
