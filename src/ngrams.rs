// src/ngrams.rs
//! N-gram extraction and hashing utilities.

use std::hash::{BuildHasherDefault, Hasher};

const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

/// FNV-1a hash of a byte slice, returning i64 (matching Python's implementation).
pub fn fnv1a_hash(bytes: &[u8]) -> i64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash as i64
}

/// Feed the little-endian bytes of one token into an FNV-1a state.
#[inline]
fn fnv1a_feed_u32(mut hash: u64, token: u32) -> u64 {
    for byte in token.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Hash an n-gram (sequence of token IDs) to an i64.
/// The n-gram is padded with a sentinel value (u32::MAX) to match Python behavior.
#[cfg_attr(not(test), allow(dead_code))]
pub fn hash_ngram(tokens: &[u32], max_ngram_length: usize) -> i64 {
    let mut padded = vec![u32::MAX; max_ngram_length];
    let len = tokens.len().min(max_ngram_length);
    padded[..len].copy_from_slice(&tokens[..len]);

    // Convert to bytes (little-endian, matching numpy's byte representation)
    let bytes: Vec<u8> = padded.iter().flat_map(|&t| t.to_le_bytes()).collect();

    fnv1a_hash(&bytes)
}

/// Call `f` with the hash of every n-gram (1-gram through max_ngram_length) in `tokens`.
///
/// Produces the same hashes as `hash_ngram` without allocating: the FNV state for each
/// prefix is extended one token at a time and the sentinel padding is fed on a copy.
pub fn for_each_ngram_hash(tokens: &[u32], max_ngram_length: usize, mut f: impl FnMut(i64)) {
    for start in 0..tokens.len() {
        let end = (start + max_ngram_length).min(tokens.len());
        let mut prefix = FNV_OFFSET_BASIS;
        for (n, &token) in tokens[start..end].iter().enumerate() {
            prefix = fnv1a_feed_u32(prefix, token);
            let mut hash = prefix;
            for _ in n + 1..max_ngram_length {
                hash = fnv1a_feed_u32(hash, u32::MAX);
            }
            f(hash as i64);
        }
    }
}

/// Extract all n-gram hashes from a token sequence (1-gram through max_ngram_length).
#[cfg_attr(not(test), allow(dead_code))]
pub fn extract_ngrams_hashed(tokens: &[u32], max_ngram_length: usize) -> Vec<i64> {
    let mut hashes = Vec::new();
    for_each_ngram_hash(tokens, max_ngram_length, |h| hashes.push(h));
    hashes
}

/// Hasher for keys that are already FNV hashes: uses the key bits directly.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 << 8) | byte as u64;
        }
    }

    fn write_i64(&mut self, n: i64) {
        self.0 = n as u64;
    }
}

pub type BuildIdentityHasher = BuildHasherDefault<IdentityHasher>;

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

    #[test]
    fn test_streaming_hashes_match_padded_hashes() {
        let tokens: Vec<u32> = vec![101, 7592, 2088, 0, u32::MAX - 1, 42, 9];
        for max_ngram_length in 1..=6 {
            let mut expected = Vec::new();
            for n in 1..=max_ngram_length {
                for window in tokens.windows(n) {
                    expected.push(hash_ngram(window, max_ngram_length));
                }
            }
            let mut actual = extract_ngrams_hashed(&tokens, max_ngram_length);
            expected.sort_unstable();
            actual.sort_unstable();
            assert_eq!(actual, expected, "max_ngram_length={max_ngram_length}");
        }
    }

    #[test]
    fn test_short_sequences() {
        assert!(extract_ngrams_hashed(&[], 5).is_empty());
        assert_eq!(extract_ngrams_hashed(&[7], 5), vec![hash_ngram(&[7], 5)]);
    }
}
