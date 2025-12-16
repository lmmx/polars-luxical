// src/tfidf.rs
//! TF-IDF transformation utilities.

use ndarray::ArrayView1;
use sprs::CsMat;

/// Apply TF-IDF weighting to a sparse BoW matrix and L2-normalize each row.
pub fn apply_tfidf_and_normalize(bow: &CsMat<f32>, idf: ArrayView1<f32>) -> CsMat<f32> {
    let (n_rows, n_cols) = (bow.rows(), bow.cols());

    let mut new_indptr = Vec::with_capacity(n_rows + 1);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    new_indptr.push(0usize);

    for row_vec in bow.outer_iterator() {
        let mut tfidf_values: Vec<(usize, f32)> = Vec::new();
        let mut norm_sq: f32 = 0.0;

        for (col_idx, &count) in row_vec.iter() {
            let idf_val = if col_idx < idf.len() { idf[col_idx] } else { 1.0 };
            let tfidf_val = count * idf_val;
            norm_sq += tfidf_val * tfidf_val;
            tfidf_values.push((col_idx, tfidf_val));
        }

        let norm = norm_sq.sqrt();
        for (col_idx, tfidf_val) in tfidf_values {
            let normalized = if norm > 0.0 { tfidf_val / norm } else { 0.0 };
            new_indices.push(col_idx);
            new_data.push(normalized);
        }

        new_indptr.push(new_indices.len());
    }

    CsMat::new((n_rows, n_cols), new_indptr, new_indices, new_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use sprs::TriMat;

    #[test]
    fn test_tfidf_normalization() {
        let mut tri = TriMat::new((2, 3));
        tri.add_triplet(0, 0, 2.0f32);
        tri.add_triplet(0, 2, 1.0);
        tri.add_triplet(1, 1, 3.0);
        let bow = tri.to_csr();

        let idf = array![1.0f32, 2.0, 0.5];
        let tfidf = apply_tfidf_and_normalize(&bow, idf.view());

        for row in tfidf.outer_iterator() {
            let norm_sq: f32 = row.iter().map(|(_, &v)| v * v).sum();
            if norm_sq > 0.0 {
                assert!((norm_sq - 1.0).abs() < 1e-6, "Row not normalized: {}", norm_sq);
            }
        }
    }
}
