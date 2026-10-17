//! # PERTARO
//!
//! **P**runed **E**xpert **R**egrounding **T**rie via **A**nswer-set **R**educt **O**ptimization.
//!
//! A `#![no_std]`, zero-heap neuro-symbolic micro-expert primitive that packs a
//! 5-tier ternary decision trie (`3^5 = 243` leaves) plus a 13-state in-band
//! control envelope (`243..=255`) into one exact 256-byte cache-aligned word.
//!
//! ## Byte contract
//!
//! One byte has 256 states: `0..243` are 5-trit coordinates, `243..=255` are
//! the 13 [`Sentinel`] control states. A [`Word256`] is header-first:
//!
//! | offset  | field          | meaning                                        |
//! |---------|----------------|------------------------------------------------|
//! | `0..8`  | `hash: u64`    | xorshift64 seed of the default leaf field       |
//! | `8..12` | `parent: u32`  | lineage link to the parent horizon word         |
//! | `12..255` | `trits[0..243]` | 243 leaf deltas or in-band sentinels         |
//! | `255`   | `trits[243]`   | root sentinel: the active horizon phase        |
//!
//! ## Default field
//!
//! The seed's xorshift64 stream is turned into 243 uniform `Z_243` digits by
//! radix-243 extraction: `243^8 < 2^64`, so every state below `243^8` holds
//! eight exact, independent digits; states at or above it are rejected, not
//! reduced. No `% 243` bias anywhere.

#![no_std]
#![forbid(unsafe_code)]

#[cfg(feature = "std")]
extern crate std;

/// Count of 5-trit coordinate values in one byte: `3^5 = 243`.
pub const MAX_PACKED: u8 = 243;

/// Count of in-band control states: `256 - 243 = 13`.
pub const SENTINEL_COUNT: usize = 256 - MAX_PACKED as usize;

/// Trie leaves: `3^5 = 243`, one byte each.
pub const LEAVES: usize = MAX_PACKED as usize;

/// Trie depth: five ternary tiers address one leaf.
pub const TIERS: usize = 5;

/// Payload index of the root sentinel byte, the one byte past the leaves.
pub const ROOT_SENTINEL: usize = LEAVES;

/// Payload width: 243 leaves + 1 root sentinel.
pub const PAYLOAD_BYTES: usize = LEAVES + 1;

/// `243^8`: the rejection threshold for unbiased radix-243 extraction.
pub const RADIX_243_POW_8: u64 = 12_157_665_459_056_928_801;

/// Digits one accepted xorshift64 state yields.
pub const DIGITS_PER_WORD: usize = 8;

/// Canonical FNV-1a 64-bit offset basis.
pub const FNV1A64_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// Canonical FNV-1a 64-bit prime.
pub const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

// ---------------------------------------------------------------------------
// ARITHMETIC LOCKS
// ---------------------------------------------------------------------------
const _: () = assert!(3usize.pow(TIERS as u32) == LEAVES);
const _: () = assert!(LEAVES + SENTINEL_COUNT == 256);
const _: () = assert!(SENTINEL_COUNT == 13);
const _: () = assert!((MAX_PACKED as u64).pow(DIGITS_PER_WORD as u32) == RADIX_243_POW_8);

/// The 13 in-band control states, bytes `243..=255`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sentinel {
    /// 243: terminal node / nominal end-of-sequence. The default prior holds.
    NullNode = 243,
    /// 244: out-of-band arithmetic overflow marker.
    MersenneOverflow = 244,
    /// 245: defeated / evicted branch under the Gelfond-Lifschitz reduct.
    Tombstone = 245,
    /// 246: bus-indirect / Tier-5 horizon regrounding trigger.
    BusIndirect = 246,
    /// 247: horizon phase 5.
    Phase5 = 247,
    /// 248: horizon phase 6.
    Phase6 = 248,
    /// 249: horizon phase 7.
    Phase7 = 249,
    /// 250: horizon phase 8.
    Phase8 = 250,
    /// 251: horizon phase 9.
    Phase9 = 251,
    /// 252: horizon phase 10.
    Phase10 = 252,
    /// 253: horizon phase 11.
    Phase11 = 253,
    /// 254: horizon phase 12.
    Phase12 = 254,
    /// 255: horizon phase 13.
    Phase13 = 255,
}

const _: () = assert!(core::mem::size_of::<Sentinel>() == 1);

impl Sentinel {
    /// The sentinel for byte `b`, or `None` when `b` is a coordinate (`< 243`).
    #[inline(always)]
    pub const fn from_u8(b: u8) -> Option<Self> {
        match b {
            243 => Some(Self::NullNode),
            244 => Some(Self::MersenneOverflow),
            245 => Some(Self::Tombstone),
            246 => Some(Self::BusIndirect),
            247 => Some(Self::Phase5),
            248 => Some(Self::Phase6),
            249 => Some(Self::Phase7),
            250 => Some(Self::Phase8),
            251 => Some(Self::Phase9),
            252 => Some(Self::Phase10),
            253 => Some(Self::Phase11),
            254 => Some(Self::Phase12),
            255 => Some(Self::Phase13),
            _ => None,
        }
    }
}

/// FNV-1a 64-bit fold of `bytes` onto a running `hash`. Start from
/// [`FNV1A64_BASIS`]. The single FNV home in this crate.
#[inline(always)]
pub const fn fnv1a64_fold(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV1A64_PRIME);
        i += 1;
    }
    hash
}

/// One xorshift64 step (`13, 7, 17`). `0` is a fixed point.
#[inline(always)]
pub const fn xorshift64(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Sealed 256-byte header-first word, aligned to 256 bytes (four cache lines).
#[repr(C, align(256))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Word256 {
    /// Deterministic PRNG seed / content hash.
    pub hash: u64,
    /// Lineage link to the parent horizon word. `0` at the head of a lineage.
    pub parent: u32,
    /// 243 leaf bytes followed by the root sentinel byte.
    pub trits: [u8; PAYLOAD_BYTES],
}

// ---------------------------------------------------------------------------
// LAYOUT LOCKS
// ---------------------------------------------------------------------------
const _: () = assert!(core::mem::size_of::<Word256>() == 256);
const _: () = assert!(core::mem::align_of::<Word256>() == 256);
const _: () = assert!(core::mem::offset_of!(Word256, hash) == 0);
const _: () = assert!(core::mem::offset_of!(Word256, parent) == 8);
const _: () = assert!(core::mem::offset_of!(Word256, trits) == 12);
const _: () = assert!(core::mem::offset_of!(Word256, trits) + ROOT_SENTINEL == 255);
const _: () = assert!(core::mem::size_of::<PertaroExpert243>() == 256);
const _: () = assert!(core::mem::align_of::<PertaroExpert243>() == 256);

impl Word256 {
    /// The all-zero word.
    pub const ZERO: Self = Self {
        hash: 0,
        parent: 0,
        trits: [0; PAYLOAD_BYTES],
    };
}

/// Zero-cost view over [`Word256`]: the payload as a 5-tier micro-expert.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PertaroExpert243(pub Word256);

impl PertaroExpert243 {
    /// The seed: the word's `hash` field.
    #[inline(always)]
    pub const fn seed(&self) -> u64 {
        self.0.hash
    }

    /// The lineage link.
    #[inline(always)]
    pub const fn parent(&self) -> u32 {
        self.0.parent
    }

    /// The raw root sentinel byte at payload index 243.
    #[inline(always)]
    pub const fn root_sentinel_byte(&self) -> u8 {
        self.0.trits[ROOT_SENTINEL]
    }

    /// The active phase, or `None` if the root byte is a coordinate.
    #[inline(always)]
    pub const fn root_sentinel(&self) -> Option<Sentinel> {
        Sentinel::from_u8(self.0.trits[ROOT_SENTINEL])
    }

    /// The 243 raw leaf bytes: deltas `0..243` or in-band sentinels.
    #[inline(always)]
    pub const fn leaves(&self) -> &[u8; LEAVES] {
        match self.0.trits.first_chunk::<LEAVES>() {
            Some(c) => c,
            None => unreachable!(),
        }
    }

    /// Leaf index `0..243` of a 5-tier ternary path. Every tier must be in
    /// `0..3`; anything else is a panic, because the path is the address.
    #[inline(always)]
    pub const fn trie_index(path: [u8; TIERS]) -> usize {
        let mut idx = 0usize;
        let mut i = 0;
        while i < TIERS {
            assert!(path[i] < 3, "pertaro trie tier outside 0..3");
            idx = idx * 3 + path[i] as usize;
            i += 1;
        }
        idx
    }

    /// Fill `out` with 243 uniform `Z_243` digits from the seed's xorshift64
    /// stream by radix-243 extraction with word rejection. A zero seed is
    /// replaced by [`FNV1A64_BASIS`] so the stream is never the zero fixed point.
    #[inline(always)]
    pub const fn default_field(seed: u64, out: &mut [u8; LEAVES]) {
        let mut s = if seed == 0 { FNV1A64_BASIS } else { seed };
        let mut idx = 0usize;
        while idx < LEAVES {
            s = xorshift64(s);
            if s >= RADIX_243_POW_8 {
                continue;
            }
            let mut word = s;
            let mut k = 0;
            while k < DIGITS_PER_WORD && idx < LEAVES {
                out[idx] = (word % MAX_PACKED as u64) as u8;
                word /= MAX_PACKED as u64;
                idx += 1;
                k += 1;
            }
        }
    }

    /// Materialize the expert: header and root byte copied verbatim, each leaf
    /// resolved as `(base + delta) mod 243`, or by its in-band sentinel
    /// (`NullNode` keeps the base; every other sentinel propagates as-is).
    #[inline(always)]
    pub const fn materialize(&self) -> Word256 {
        let mut base = [0u8; LEAVES];
        Self::default_field(self.0.hash, &mut base);
        let mut out = self.0;
        let mut idx = 0usize;
        while idx < LEAVES {
            let raw = self.0.trits[idx];
            out.trits[idx] = if raw < MAX_PACKED {
                ((base[idx] as u16 + raw as u16) % MAX_PACKED as u16) as u8
            } else if raw == Sentinel::NullNode as u8 {
                base[idx]
            } else {
                raw
            };
            idx += 1;
        }
        out
    }

    /// Reground at the Tier-5 boundary: fold the seed, the winning leaf
    /// (`0..243`) and the next phase through FNV-1a, then one xorshift64 step.
    /// A winner at or above 243 is a panic: sentinels are not leaves.
    #[inline(always)]
    pub const fn reground_seed(&self, winner: u8, next: Sentinel) -> u64 {
        assert!(
            winner < MAX_PACKED,
            "pertaro reground winner outside the 243 leaves"
        );
        let h = fnv1a64_fold(FNV1A64_BASIS, &self.0.hash.to_le_bytes());
        let h = fnv1a64_fold(h, &[winner, next as u8]);
        xorshift64(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(seed: u64) -> Word256 {
        Word256 {
            hash: seed,
            parent: 7,
            trits: [0; PAYLOAD_BYTES],
        }
    }

    #[test]
    fn every_byte_is_a_coordinate_or_a_sentinel() {
        for b in 0..MAX_PACKED {
            assert!(Sentinel::from_u8(b).is_none(), "{b}");
        }
        for b in MAX_PACKED..=255 {
            assert_eq!(Sentinel::from_u8(b).map(|s| s as u8), Some(b));
        }
    }

    #[test]
    fn default_field_is_deterministic_and_in_range() {
        let mut a = [0u8; LEAVES];
        let mut b = [0u8; LEAVES];
        PertaroExpert243::default_field(0x1234_5678_9ABC_DEF0, &mut a);
        PertaroExpert243::default_field(0x1234_5678_9ABC_DEF0, &mut b);
        assert_eq!(a, b);
        assert!(a.iter().all(|x| *x < MAX_PACKED));
        let mut c = [0u8; LEAVES];
        PertaroExpert243::default_field(0x1234_5678_9ABC_DEF1, &mut c);
        assert_ne!(a, c);
    }

    #[test]
    fn zero_seed_uses_the_basis() {
        let mut z = [0u8; LEAVES];
        let mut b = [0u8; LEAVES];
        PertaroExpert243::default_field(0, &mut z);
        PertaroExpert243::default_field(FNV1A64_BASIS, &mut b);
        assert_eq!(z, b);
    }

    #[test]
    fn radix_digits_are_balanced_over_many_seeds() {
        // 2000 seeds x 243 digits: every residue class must appear, and the
        // low 13 classes (the ones `% 243` would double) must not dominate.
        let mut hist = [0u32; 243];
        let mut f = [0u8; LEAVES];
        for seed in 1..=2000u64 {
            PertaroExpert243::default_field(seed.wrapping_mul(FNV1A64_PRIME), &mut f);
            for d in f {
                hist[d as usize] += 1;
            }
        }
        let total: u32 = hist.iter().sum();
        let mean = total / 243;
        for (d, n) in hist.iter().enumerate() {
            assert!(
                *n > mean / 2 && *n < mean * 2,
                "digit {d}: {n} vs mean {mean}"
            );
        }
    }

    #[test]
    fn materialize_keeps_header_and_resolves_sentinels() {
        let mut w = word(99);
        w.trits[0] = 5;
        w.trits[1] = Sentinel::NullNode as u8;
        w.trits[2] = Sentinel::Tombstone as u8;
        w.trits[3] = Sentinel::BusIndirect as u8;
        w.trits[ROOT_SENTINEL] = Sentinel::Phase13 as u8;
        let x = PertaroExpert243(w);
        let out = x.materialize();
        let mut base = [0u8; LEAVES];
        PertaroExpert243::default_field(99, &mut base);
        assert_eq!(out.hash, 99);
        assert_eq!(out.parent, 7);
        assert_eq!(out.trits[0], ((base[0] as u16 + 5) % 243) as u8);
        assert_eq!(out.trits[1], base[1]);
        assert_eq!(out.trits[2], Sentinel::Tombstone as u8);
        assert_eq!(out.trits[3], Sentinel::BusIndirect as u8);
        assert_eq!(out.trits[ROOT_SENTINEL], Sentinel::Phase13 as u8);
        assert_eq!(x.root_sentinel(), Some(Sentinel::Phase13));
        assert_eq!(x.leaves()[0], 5);
    }

    #[test]
    fn trie_index_spans_all_leaves() {
        assert_eq!(PertaroExpert243::trie_index([0; 5]), 0);
        assert_eq!(PertaroExpert243::trie_index([2; 5]), LEAVES - 1);
        assert_eq!(PertaroExpert243::trie_index([1, 0, 0, 0, 0]), 81);
        assert_eq!(PertaroExpert243::trie_index([0, 0, 0, 0, 1]), 1);
    }

    #[test]
    #[should_panic]
    fn trie_index_refuses_a_bad_tier() {
        let _ = PertaroExpert243::trie_index([0, 0, 3, 0, 0]);
    }

    #[test]
    fn reground_is_deterministic_and_phase_sensitive() {
        let x = PertaroExpert243(word(1));
        let a = x.reground_seed(0, Sentinel::NullNode);
        assert_eq!(a, x.reground_seed(0, Sentinel::NullNode));
        assert_ne!(a, x.reground_seed(1, Sentinel::NullNode));
        assert_ne!(a, x.reground_seed(0, Sentinel::Tombstone));
        assert_ne!(a, 0);
    }

    #[test]
    #[should_panic]
    fn reground_refuses_a_sentinel_as_winner() {
        let _ = PertaroExpert243(word(1)).reground_seed(243, Sentinel::NullNode);
    }
}
