// src/embedder/safetensors_loader.rs
//! Safetensors format loader for Luxical models - supports memory-mapped loading.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use super::model::{LuxicalEmbedder, LuxicalError, Result};

/// Load a Luxical model from a safetensors file using memory mapping.
pub fn load_from_safetensors<P: AsRef<Path>>(path: P) -> Result<LuxicalEmbedder> {
    let path = path.as_ref();
    let file = File::open(path)?;

    // Memory-map the file for fast access
    let mmap = unsafe { Mmap::map(&file)? };

    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| LuxicalError::Safetensors(e.to_string()))?;

    // Check version
    let version_tensor = tensors
        .tensor("embedder.version")
        .map_err(|e| LuxicalError::Safetensors(format!("Missing version: {}", e)))?;
    let version_data: &[i64] = bytemuck::cast_slice(version_tensor.data());
    if version_data.is_empty() || version_data[0] != 1 {
        return Err(LuxicalError::Safetensors(format!(
            "Unsupported version: {:?}",
            version_data
        )));
    }

    // Load tokenizer
    let tokenizer_tensor = tensors
        .tensor("embedder.tokenizer")
        .map_err(|e| LuxicalError::Safetensors(format!("Missing tokenizer: {}", e)))?;
    let tokenizer_bytes: &[u8] = tokenizer_tensor.data();
    let tokenizer_json = std::str::from_utf8(tokenizer_bytes)
        .map_err(|e| LuxicalError::Parse(format!("Invalid tokenizer UTF-8: {}", e)))?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
        .map_err(|e| LuxicalError::Tokenizer(e.to_string()))?;

    // Load recognized ngrams to get max_ngram_length
    let ngrams_tensor = tensors
        .tensor("embedder.recognized_ngrams")
        .map_err(|e| LuxicalError::Safetensors(format!("Missing recognized_ngrams: {}", e)))?;
    let ngrams_shape = ngrams_tensor.shape();
    let max_ngram_length = if ngrams_shape.len() == 2 {
        ngrams_shape[1]
    } else {
        5
    };

    // Load ngram hash map
    let keys_tensor = tensors
        .tensor("embedder.ngram_keys")
        .map_err(|e| LuxicalError::Safetensors(format!("Missing ngram_keys: {}", e)))?;
    let vals_tensor = tensors
        .tensor("embedder.ngram_vals")
        .map_err(|e| LuxicalError::Safetensors(format!("Missing ngram_vals: {}", e)))?;

    let keys: &[i64] = bytemuck::cast_slice(keys_tensor.data());
    let vals: &[i64] = bytemuck::cast_slice(vals_tensor.data());

    let mut ngram_hash_to_idx: HashMap<i64, u32> = HashMap::with_capacity(keys.len());
    ngram_hash_to_idx.extend(keys.iter().copied().zip(vals.iter().map(|&v| v as u32)));

    // Load IDF values
    let idf_tensor = tensors
        .tensor("embedder.idf_values")
        .map_err(|e| LuxicalError::Safetensors(format!("Missing idf_values: {}", e)))?;
    let idf_data: &[f32] = bytemuck::cast_slice(idf_tensor.data());
    let idf_values = Array1::from_vec(idf_data.to_vec());

    // Load neural network layers
    let num_layers_tensor = tensors
        .tensor("embedder.num_layers")
        .map_err(|e| LuxicalError::Safetensors(format!("Missing num_layers: {}", e)))?;
    let num_layers_data: &[i64] = bytemuck::cast_slice(num_layers_tensor.data());
    let num_layers = num_layers_data[0] as usize;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let layer_name = format!("embedder.nn_layer_{}", i);
        let layer_tensor = tensors
            .tensor(&layer_name)
            .map_err(|e| LuxicalError::Safetensors(format!("Missing {}: {}", layer_name, e)))?;

        let shape = layer_tensor.shape();
        if shape.len() != 2 {
            return Err(LuxicalError::Safetensors(format!(
                "Expected 2D layer, got {}D",
                shape.len()
            )));
        }

        let layer_data: &[f32] = bytemuck::cast_slice(layer_tensor.data());
        let layer = Array2::from_shape_vec((shape[0], shape[1]), layer_data.to_vec())
            .map_err(|e| LuxicalError::Parse(e.to_string()))?;
        layers.push(layer);
    }

    let output_dim = layers.last().map(|l| l.nrows()).unwrap_or(0);

    Ok(LuxicalEmbedder {
        tokenizer,
        max_ngram_length,
        ngram_hash_to_idx,
        idf_values,
        layers,
        output_dim,
    })
}