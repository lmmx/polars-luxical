// src/ngrams.rs
//! N-gram extraction and hashing utilities.

/// FNV-1a hash of a byte slice, returning i64 (matching Python's implementation).
pub fn fnv1a_hash(bytes: &[u8]) -> i64 {
    const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash as i64
}

/// Hash an n-gram (sequence of token IDs) to an i64.
/// The n-gram is padded with a sentinel value (u32::MAX) to match Python behavior.
pub fn hash_ngram(tokens: &[u32], max_ngram_length: usize) -> i64 {
    let mut padded = vec![u32::MAX; max_ngram_length];
    let len = tokens.len().min(max_ngram_length);
    padded[..len].copy_from_slice(&tokens[..len]);

    // Convert to bytes (little-endian, matching numpy's byte representation)
    let bytes: Vec<u8> = padded.iter().flat_map(|&t| t.to_le_bytes()).collect();

    fnv1a_hash(&bytes)
}

/// Extract all n-gram hashes from a token sequence (1-gram through max_ngram_length).
pub fn extract_ngrams_hashed(tokens: &[u32], max_ngram_length: usize) -> Vec<i64> {
    let mut hashes = Vec::new();

    for ngram_len in 1..=max_ngram_length {
        if tokens.len() < ngram_len {
            continue;
        }
        for start in 0..=(tokens.len() - ngram_len) {
            let ngram = &tokens[start..start + ngram_len];
            hashes.push(hash_ngram(ngram, max_ngram_length));
        }
    }

    hashes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_hash_deterministic() {
        let bytes = b"hello world";
        let hash1 = fnv1a_hash(bytes);
        let hash2 = fnv1a_hash(bytes);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_hash_ngram_padding() {
        let tokens = vec![1u32, 2, 3];
        let hash1 = hash_ngram(&tokens[..2], 5);
        let hash2 = hash_ngram(&tokens[..2], 5);
        assert_eq!(hash1, hash2);

        let hash3 = hash_ngram(&tokens[..3], 5);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_extract_ngrams() {
        let tokens = vec![1u32, 2, 3, 4];
        let hashes = extract_ngrams_hashed(&tokens, 2);
        // Should have: 4 unigrams + 3 bigrams = 7 n-grams
        assert_eq!(hashes.len(), 7);
    }
}
