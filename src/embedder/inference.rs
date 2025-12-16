// src/embedder/inference.rs
//! Inference logic for the Luxical embedder.

use std::collections::HashMap;

use ndarray::{Array2, Axis};
use sprs::{CsMat, TriMat};

use super::model::{LuxicalEmbedder, Result};
use crate::ngrams::extract_ngrams_hashed;
use crate::tfidf::apply_tfidf_and_normalize;

impl LuxicalEmbedder {
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
                    .map_err(|e| super::model::LuxicalError::Tokenizer(e.to_string()))?;
                Ok(encoding.get_ids().to_vec())
            })
            .collect::<Result<Vec<_>>>()?;

        // 2. Build sparse BoW matrix with n-gram hashing
        let bow = self.build_bow_matrix(&token_ids);

        // 3. Apply TF-IDF weighting
        let tfidf = apply_tfidf_and_normalize(&bow, self.idf_values.view());

        // 4. Forward pass through MLP
        let embeddings = self.forward(&tfidf);

        Ok(embeddings)
    }

    /// Build a sparse BoW (bag-of-words) matrix from tokenized documents.
    fn build_bow_matrix(&self, token_ids: &[Vec<u32>]) -> CsMat<f32> {
        let n_docs = token_ids.len();
        let vocab_size = self.vocab_size();

        let mut triplets = TriMat::new((n_docs, vocab_size));

        for (doc_idx, tokens) in token_ids.iter().enumerate() {
            let mut counts: HashMap<u32, u32> = HashMap::new();

            for ngram_hash in extract_ngrams_hashed(tokens, self.max_ngram_length) {
                if let Some(&vocab_idx) = self.ngram_hash_to_idx.get(&ngram_hash) {
                    *counts.entry(vocab_idx).or_insert(0) += 1;
                }
            }

            for (vocab_idx, count) in counts {
                triplets.add_triplet(doc_idx, vocab_idx as usize, count as f32);
            }
        }

        triplets.to_csr()
    }

    /// Forward pass through the MLP layers.
    fn forward(&self, tfidf: &CsMat<f32>) -> Array2<f32> {
        debug_assert!(!self.layers.is_empty());

        // First layer: sparse @ dense.T
        let mut hidden = sparse_dense_matmul(tfidf, &self.layers[0]);
        relu_inplace(&mut hidden);
        normalize_rows_inplace(&mut hidden);

        // Hidden layers
        for layer in &self.layers[1..self.layers.len() - 1] {
            hidden = hidden.dot(&layer.t());
            relu_inplace(&mut hidden);
            normalize_rows_inplace(&mut hidden);
        }

        // Final layer (no ReLU)
        if self.layers.len() > 1 {
            hidden = hidden.dot(&self.layers.last().unwrap().t());
            normalize_rows_inplace(&mut hidden);
        }

        hidden
    }
}

/// Sparse matrix (CSR) times dense matrix (stored as output_dim x input_dim).
fn sparse_dense_matmul(sparse: &CsMat<f32>, dense: &Array2<f32>) -> Array2<f32> {
    let n_rows = sparse.rows();
    let output_dim = dense.nrows();

    let mut result = Array2::zeros((n_rows, output_dim));

    for (row_idx, row_vec) in sparse.outer_iterator().enumerate() {
        for (col_idx, &val) in row_vec.iter() {
            let dense_col = dense.column(col_idx);
            for (out_idx, &dense_val) in dense_col.iter().enumerate() {
                result[[row_idx, out_idx]] += val * dense_val;
            }
        }
    }

    result
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