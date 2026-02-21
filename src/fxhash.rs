//! FxHash — fast non-cryptographic hasher for internal data (~3ns vs SipHash ~20ns).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x517cc1b727220a95;

pub struct FxHasher { hash: u64 }

impl Default for FxHasher {
    #[inline] fn default() -> Self { Self { hash: 0 } }
}

impl Hasher for FxHasher {
    #[inline] fn finish(&self) -> u64 { self.hash }
    #[inline] fn write(&mut self, bytes: &[u8]) {
        let mut i = 0;
        let len = bytes.len();
        while i + 8 <= len {
            let word = u64::from_ne_bytes(unsafe {
                *(bytes.as_ptr().add(i) as *const [u8; 8])
            });
            self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
            i += 8;
        }
        while i < len {
            self.hash = (self.hash.rotate_left(5) ^ bytes[i] as u64).wrapping_mul(SEED);
            i += 1;
        }
    }
    #[inline] fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(SEED);
    }
    #[inline] fn write_usize(&mut self, i: usize) {
        self.hash = (self.hash.rotate_left(5) ^ i as u64).wrapping_mul(SEED);
    }
}

pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
pub type FxHashSet<T> = HashSet<T, FxBuildHasher>;

#[inline]
pub fn map_with_capacity<K, V>(cap: usize) -> FxHashMap<K, V> {
    HashMap::with_capacity_and_hasher(cap, FxBuildHasher::default())
}

#[inline]
pub fn set_with_capacity<T>(cap: usize) -> FxHashSet<T> {
    HashSet::with_capacity_and_hasher(cap, FxBuildHasher::default())
}
