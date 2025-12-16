// src/expressions.rs
//! Polars expressions for Luxical embeddings.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::registry::get_or_load_model;

#[derive(Deserialize)]
pub struct EmbedTextKwargs {
    #[serde(default)]
    pub model_id: Option<String>,
}

fn output_type_func(input_fields: &[Field]) -> PolarsResult<Field> {
    let embedder = get_or_load_model(&None)?;
    let dim = embedder.output_dim();

    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::Array(Box::new(DataType::Float32), dim),
    ))
}

#[polars_expr(output_type_func=output_type_func)]
pub fn embed_text(inputs: &[Series], kwargs: EmbedTextKwargs) -> PolarsResult<Series> {
    let s = &inputs[0];

    if s.dtype() != &DataType::String {
        polars_bail!(InvalidOperation:
            format!("Data type {:?} not supported. Must be a String column.", s.dtype())
        );
    }

    let embedder = get_or_load_model(&kwargs.model_id)?;
    let dim = embedder.output_dim();

    let ca = s.str()?;

    let mut texts: Vec<&str> = Vec::with_capacity(ca.len());
    let mut text_indices: Vec<usize> = Vec::with_capacity(ca.len());

    for (idx, opt_str) in ca.into_iter().enumerate() {
        if let Some(text) = opt_str {
            texts.push(text);
            text_indices.push(idx);
        }
    }

    let embeddings = if !texts.is_empty() {
        embedder
            .embed_batch(&texts)
            .map_err(|e| PolarsError::ComputeError(format!("Embedding failed: {}", e).into()))?
    } else {
        ndarray::Array2::zeros((0, dim))
    };

    let mut row_embeddings: Vec<Option<Vec<f32>>> = vec![None; ca.len()];
    for (emb_idx, &orig_idx) in text_indices.iter().enumerate() {
        let emb_row = embeddings.row(emb_idx);
        row_embeddings[orig_idx] = Some(emb_row.to_vec());
    }

    use polars::chunked_array::builder::ListPrimitiveChunkedBuilder;

    let mut builder = ListPrimitiveChunkedBuilder::<Float32Type>::new(
        s.name().clone(),
        row_embeddings.len(),
        row_embeddings.len() * dim,
        DataType::Float32,
    );

    for opt_vec in &row_embeddings {
        match opt_vec {
            Some(v) => builder.append_slice(v),
            None => builder.append_null(),
        }
    }

    let list_series = builder.finish().into_series();
    let array_series = list_series.cast(&DataType::Array(Box::new(DataType::Float32), dim))?;

    Ok(array_series)
}