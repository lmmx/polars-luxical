// src/embedder/npz_loader.rs
//! NPZ format loader for Luxical models.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use ndarray::{Array1, Array2};
use tokenizers::Tokenizer;

use super::model::{LuxicalEmbedder, LuxicalError, Result};

/// Load a Luxical model from an NPZ file.
pub fn load_from_npz<P: AsRef<Path>>(path: P) -> Result<LuxicalEmbedder> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(1024 * 1024, file);
    load_from_reader(reader)
}

fn load_from_reader<R: Read + Seek>(reader: R) -> Result<LuxicalEmbedder> {
    let mut npz = NpzReader::new(reader)?;

    // Load tokenizer from JSON bytes
    let tokenizer_bytes = npz.read_array_u8("tokenizer")?;
    let tokenizer_json = String::from_utf8(tokenizer_bytes)
        .map_err(|e| LuxicalError::Parse(format!("Invalid UTF-8 in tokenizer: {}", e)))?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
        .map_err(|e| LuxicalError::Tokenizer(e.to_string()))?;

    // Load recognized ngrams to determine max_ngram_length
    let (_, recognized_ngrams_shape) = npz.read_array_u32("recognized_ngrams")?;
    let max_ngram_length = if recognized_ngrams_shape.len() == 2 {
        recognized_ngrams_shape[1]
    } else {
        5
    };

    // Build ngram hash to index map
    let (keys, _) = npz.read_array_i64("ngram_hash_to_ngram_idx_keys")?;
    let (values, _) = npz.read_array_u32("ngram_hash_to_ngram_idx_values")?;

    let mut ngram_hash_to_idx: HashMap<i64, u32> = HashMap::with_capacity(keys.len());
    ngram_hash_to_idx.extend(keys.into_iter().zip(values.into_iter()));

    // Load IDF values
    let (idf_values_vec, _) = npz.read_array_f32("idf_values")?;
    let idf_values = Array1::from_vec(idf_values_vec);

    // Load neural network layers
    let (num_layers_vec, _) = npz.read_array_i64("num_nn_layers")?;
    let num_layers = num_layers_vec[0] as usize;

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        let (layer_vec, layer_shape) = npz.read_array_f32(&format!("nn_layer_{}", i))?;
        if layer_shape.len() != 2 {
            return Err(LuxicalError::Parse(format!(
                "Expected 2D layer, got {}D",
                layer_shape.len()
            )));
        }
        let layer = Array2::from_shape_vec((layer_shape[0], layer_shape[1]), layer_vec)
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

// ============================================================================
// NPZ Reader
// ============================================================================

struct NpzReader<R: Read + Seek> {
    archive: zip::ZipArchive<R>,
}

impl<R: Read + Seek> NpzReader<R> {
    fn new(reader: R) -> Result<Self> {
        let archive =
            zip::ZipArchive::new(reader).map_err(|e| LuxicalError::Parse(e.to_string()))?;
        Ok(Self { archive })
    }

    fn read_npy_header(data: &[u8]) -> Result<(Vec<usize>, String, usize)> {
        if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
            return Err(LuxicalError::Parse("Invalid NPY magic".to_string()));
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
            .map_err(|e| LuxicalError::Parse(format!("Invalid header UTF-8: {}", e)))?;

        let shape = Self::parse_shape(header_str)?;
        let dtype = Self::parse_dtype(header_str)?;

        Ok((shape, dtype, header_end))
    }

    fn parse_shape(header: &str) -> Result<Vec<usize>> {
        let shape_marker = "'shape':";
        let shape_start = header
            .find(shape_marker)
            .ok_or_else(|| LuxicalError::Parse("No shape in header".to_string()))?;
        let rest = &header[shape_start + shape_marker.len()..];
        let rest = rest.trim_start();

        let paren_start = rest
            .find('(')
            .ok_or_else(|| LuxicalError::Parse("No shape tuple".to_string()))?;
        let paren_end = rest
            .find(')')
            .ok_or_else(|| LuxicalError::Parse("No shape tuple end".to_string()))?;
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
            .ok_or_else(|| LuxicalError::Parse("No descr in header".to_string()))?;
        let rest = &header[descr_start + descr_marker.len()..];
        let rest = rest.trim_start();

        if !rest.starts_with('\'') {
            return Err(LuxicalError::Parse(format!(
                "Expected quote after 'descr':', got: {}",
                &rest[..rest.len().min(20)]
            )));
        }
        let rest = &rest[1..];

        let quote_end = rest
            .find('\'')
            .ok_or_else(|| LuxicalError::Parse("No descr end quote".to_string()))?;
        Ok(rest[..quote_end].to_string())
    }

    fn read_raw_array(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, String)> {
        let npy_name = format!("{}.npy", name);
        let mut file = self
            .archive
            .by_name(&npy_name)
            .map_err(|_| LuxicalError::Parse(format!("Array '{}' not found", name)))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let (shape, dtype, data_start) = Self::read_npy_header(&data)?;
        Ok((data[data_start..].to_vec(), shape, dtype))
    }

    fn read_array_u8(&mut self, name: &str) -> Result<Vec<u8>> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("u1") && !dtype.contains("uint8") {
            return Err(LuxicalError::Parse(format!("Expected uint8, got {}", dtype)));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        Ok(data[..n_elements].to_vec())
    }

    fn read_array_f32(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("f4") && !dtype.contains("float32") {
            return Err(LuxicalError::Parse(format!("Expected float32, got {}", dtype)));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = n_elements * 4;

        let result: Vec<f32> = bytemuck::cast_slice(&data[..expected_bytes]).to_vec();
        Ok((result, shape))
    }

    fn read_array_i64(&mut self, name: &str) -> Result<(Vec<i64>, Vec<usize>)> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("i8") && !dtype.contains("int64") {
            return Err(LuxicalError::Parse(format!("Expected int64, got {}", dtype)));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = n_elements * 8;

        let result: Vec<i64> = bytemuck::cast_slice(&data[..expected_bytes]).to_vec();
        Ok((result, shape))
    }

    fn read_array_u32(&mut self, name: &str) -> Result<(Vec<u32>, Vec<usize>)> {
        let (data, shape, dtype) = self.read_raw_array(name)?;

        if !dtype.contains("u4") && !dtype.contains("uint32") {
            return Err(LuxicalError::Parse(format!("Expected uint32, got {}", dtype)));
        }

        let n_elements: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        let expected_bytes = n_elements * 4;

        let result: Vec<u32> = bytemuck::cast_slice(&data[..expected_bytes]).to_vec();
        Ok((result, shape))
    }
}
