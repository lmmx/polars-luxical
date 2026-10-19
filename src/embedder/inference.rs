// src/embedder/inference.rs
//! Inference logic for the Luxical embedder.

use ndarray::{concatenate, s, Array2, ArrayView2, Axis};
use rayon::prelude::*;

use super::model::{LuxicalEmbedder, LuxicalError, Result};
use crate::ngrams::for_each_ngram_hash;

/// Rows per chunk when running the dense layers in parallel.
const DENSE_CHUNK_ROWS: usize = 256;

impl LuxicalEmbedder {
    /// Embed a batch of texts, returning a 2D array of shape (n_texts, output_dim).
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Array2<f32>> {
        if texts.is_empty() {
            return Ok(Array2::zeros((0, self.output_dim)));
        }

        // 1. Per document, in parallel: tokenize, count n-grams, TF-IDF weight and
        //    project through the first (sparse) layer
        let hidden_dim = self.first_layer_t.ncols();
        let mut hidden = vec![0f32; texts.len() * hidden_dim];
        hidden
            .par_chunks_mut(hidden_dim)
            .zip(texts.par_iter())
            .try_for_each(|(row, text)| -> Result<()> {
                let encoding = self
                    .tokenizer
                    .encode(*text, false)
                    .map_err(|e| LuxicalError::Tokenizer(e.to_string()))?;
                self.first_layer_row(encoding.get_ids(), row);
                Ok(())
            })?;
        let hidden = Array2::from_shape_vec((texts.len(), hidden_dim), hidden)
            .map_err(|e| LuxicalError::Parse(e.to_string()))?;

        // 2. Dense layers, in parallel over row chunks
        Ok(self.forward_dense(hidden.view()))
    }

    /// Compute one document's first-layer activation (after ReLU and L2 norm) into `out`.
    fn first_layer_row(&self, tokens: &[u32], out: &mut [f32]) {
        // Vocabulary indices of all recognised n-grams, sorted so equal indices are adjacent
        let mut indices: Vec<u32> = Vec::new();
        for_each_ngram_hash(tokens, self.max_ngram_length, |hash| {
            if let Some(&vocab_idx) = self.ngram_hash_to_idx.get(&hash) {
                indices.push(vocab_idx);
            }
        });
        indices.sort_unstable();

        // TF-IDF weights (count * idf) per distinct index, then L2 normalise
        let mut weights: Vec<(usize, f32)> = Vec::new();
        for group in indices.chunk_by(|a, b| a == b) {
            let idx = group[0] as usize;
            let idf = self.idf_values.get(idx).copied().unwrap_or(1.0);
            weights.push((idx, group.len() as f32 * idf));
        }
        let norm = weights.iter().map(|&(_, w)| w * w).sum::<f32>().sqrt();
        if norm == 0.0 {
            return;
        }

        // Weighted sum of the n-grams' rows of the transposed first layer
        for (idx, w) in weights {
            let w = w / norm;
            for (o, &x) in out.iter_mut().zip(self.first_layer_t.row(idx)) {
                *o += w * x;
            }
        }
        relu_inplace(out);
        normalize_inplace(out);
    }

    /// Run the dense layers: ReLU and L2 norm after each hidden layer, L2 norm only after
    /// the final layer.
    fn forward_dense(&self, hidden: ArrayView2<f32>) -> Array2<f32> {
        if self.dense_layers.is_empty() {
            return hidden.to_owned();
        }
        let n_rows = hidden.nrows();
        let chunks: Vec<Array2<f32>> = (0..n_rows.div_ceil(DENSE_CHUNK_ROWS))
            .into_par_iter()
            .map(|c| {
                let start = c * DENSE_CHUNK_ROWS;
                let end = (start + DENSE_CHUNK_ROWS).min(n_rows);
                let mut h = hidden.slice(s![start..end, ..]).to_owned();
                let last = self.dense_layers.len() - 1;
                for (i, layer) in self.dense_layers.iter().enumerate() {
                    h = h.dot(&layer.t());
                    for mut row in h.axis_iter_mut(Axis(0)) {
                        let row = row.as_slice_mut().expect("standard layout");
                        if i < last {
                            relu_inplace(row);
                        }
                        normalize_inplace(row);
                    }
                }
                h
            })
            .collect();
        let views: Vec<ArrayView2<f32>> = chunks.iter().map(|c| c.view()).collect();
        concatenate(Axis(0), &views).expect("chunks share column count")
    }
}

/// Apply ReLU in-place.
fn relu_inplace(row: &mut [f32]) {
    for x in row.iter_mut() {
        *x = x.max(0.0);
    }
}

/// L2 normalize in-place (rows with zero norm are left unchanged).
fn normalize_inplace(row: &mut [f32]) {
    let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
}
