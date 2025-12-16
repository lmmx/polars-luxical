// src/embedder/mod.rs
//! Luxical embedder module - handles model loading and inference.

mod inference;
mod model;
mod npz_loader;
mod safetensors_loader;

pub use model::{LuxicalEmbedder, LuxicalError, Result};
