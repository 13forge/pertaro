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
#![doc = include_str!("../README.md")]

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

// ---------------------------------------------------------------------------
// 121-NODE L1 ARENA AND THE GELFOND-LIFSCHITZ REDUCT
// ---------------------------------------------------------------------------

/// Tiers resident in the arena: `0..=4`.
pub const ARENA_TIERS: usize = 5;

/// Nodes in a complete 4-tier ternary routing arena:
/// `1 + 3 + 9 + 27 + 81 = 121`.
pub const ARENA_NODES: usize = (3usize.pow(ARENA_TIERS as u32) - 1) / 2;

/// Index of the first Tier-4 frontier node: `1 + 3 + 9 + 27 = 40`.
pub const FRONTIER_START: usize = (3usize.pow(ARENA_TIERS as u32 - 1) - 1) / 2;

/// Frontier nodes: `3^4 = 81`.
pub const FRONTIER_NODES: usize = ARENA_NODES - FRONTIER_START;

/// Exact byte footprint of the arena: `121 * 256 = 30,976` bytes.
pub const ARENA_BYTES: usize = ARENA_NODES * core::mem::size_of::<Word256>();

/// The L1 data cache size the arena is sized against.
pub const L1_BUDGET_BYTES: usize = 32 * 1024;

const _: () = assert!(ARENA_NODES == 121);
const _: () = assert!(FRONTIER_START == 40);
const _: () = assert!(FRONTIER_NODES == 81);
const _: () = assert!(ARENA_BYTES == 30_976);
const _: () = assert!(ARENA_BYTES < L1_BUDGET_BYTES);
// Tier 5 (364 nodes) does not fit: the frontier is a hardware fact, not a tunable.
const _: () = assert!((ARENA_NODES + FRONTIER_NODES * 3) * 256 > L1_BUDGET_BYTES);

/// A 256-bit set over atoms `0..=255`: a candidate or derived answer set
/// `S` over the 243 leaves. Four `u64` words, no heap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AnswerSetMask(pub [u64; 4]);

impl AnswerSetMask {
    /// The empty set.
    #[inline(always)]
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    /// Insert `atom`.
    #[inline(always)]
    pub const fn insert(&mut self, atom: u8) {
        self.0[(atom >> 6) as usize] |= 1u64 << (atom & 63);
    }

    /// Whether `atom` is in the set.
    #[inline(always)]
    pub const fn contains(&self, atom: u8) -> bool {
        (self.0[(atom >> 6) as usize] & (1u64 << (atom & 63))) != 0
    }

    /// Number of atoms in the set.
    #[inline(always)]
    pub const fn count_ones(&self) -> u32 {
        self.0[0].count_ones()
            + self.0[1].count_ones()
            + self.0[2].count_ones()
            + self.0[3].count_ones()
    }

    /// Whether the set is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.0[0] | self.0[1] | self.0[2] | self.0[3] == 0
    }

    /// The lowest atom in the set that is a leaf (`< 243`), or `None`.
    #[inline(always)]
    pub const fn first_winner(&self) -> Option<u8> {
        let mut w = 0;
        while w < 4 {
            if self.0[w] != 0 {
                let atom = (w as u8) * 64 + self.0[w].trailing_zeros() as u8;
                return if atom < MAX_PACKED { Some(atom) } else { None };
            }
            w += 1;
        }
        None
    }
}

/// The complete 4-tier L1-resident ternary trie arena: 121 words, 30,976 bytes.
///
/// Breadth-first layout: tier `t` occupies `(3^t - 1) / 2 .. (3^(t+1) - 1) / 2`.
///
/// | tier | indices    | nodes |
/// |------|------------|-------|
/// | 0    | `0`        | 1     |
/// | 1    | `1..4`     | 3     |
/// | 2    | `4..13`    | 9     |
/// | 3    | `13..40`   | 27    |
/// | 4    | `40..121`  | 81 (the frontier) |
#[repr(C, align(256))]
#[derive(Clone, Copy)]
pub struct PertaroArena121 {
    /// The nodes, breadth-first.
    pub nodes: [PertaroExpert243; ARENA_NODES],
}

const _: () = assert!(core::mem::size_of::<PertaroArena121>() == ARENA_BYTES);
const _: () = assert!(core::mem::align_of::<PertaroArena121>() == 256);

impl PertaroArena121 {
    /// An arena of all-zero words.
    pub const ZERO: Self = Self {
        nodes: [PertaroExpert243(Word256::ZERO); ARENA_NODES],
    };

    /// Flat index of the node at `tier` (`0..=4`), position `offset` (`0..3^tier`).
    /// Out-of-range input is a panic: the address is the contract.
    #[inline(always)]
    pub const fn node_index(tier: usize, offset: usize) -> usize {
        const TIER_BASE: [usize; ARENA_TIERS] = [0, 1, 4, 13, 40];
        assert!(tier < ARENA_TIERS, "pertaro arena tier outside 0..5");
        assert!(
            offset < 3usize.pow(tier as u32),
            "pertaro arena offset outside the tier"
        );
        TIER_BASE[tier] + offset
    }

    /// The three child indices of the node at `parent` (`0..40`).
    #[inline(always)]
    pub const fn children_of(parent: usize) -> [usize; 3] {
        assert!(
            parent < FRONTIER_START,
            "pertaro arena: a frontier node has no children"
        );
        let base = 3 * parent + 1;
        [base, base + 1, base + 2]
    }

    /// The parent index of the node at `idx` (`1..121`).
    #[inline(always)]
    pub const fn parent_of(idx: usize) -> usize {
        assert!(
            idx > 0 && idx < ARENA_NODES,
            "pertaro arena: index outside 1..121"
        );
        (idx - 1) / 3
    }

    /// Expand a root seed across all 121 nodes on `input_atom` (`0..243`).
    ///
    /// Every non-root node's seed is derived from its parent by
    /// [`PertaroExpert243::reground_seed`] on `(input_atom + branch) mod 243`,
    /// its `parent` field is set to the parent's index, and its root byte to
    /// `phase`. Leaf bytes are untouched: they are the stored exceptions, seeds
    /// are the derived defaults. An `input_atom` at or above 243 is a panic.
    pub fn expand_from_root(&mut self, root_seed: u64, input_atom: u8, phase: Sentinel) {
        assert!(
            input_atom < MAX_PACKED,
            "pertaro expand: input atom outside the 243 leaves"
        );
        self.nodes[0].0.hash = root_seed;
        self.nodes[0].0.parent = 0;
        self.nodes[0].0.trits[ROOT_SENTINEL] = phase as u8;

        let mut parent = 0usize;
        while parent < FRONTIER_START {
            let children = Self::children_of(parent);
            let mut seeds = [0u64; 3];
            let mut b = 0usize;
            while b < 3 {
                let winner = ((input_atom as usize + b) % LEAVES) as u8;
                seeds[b] = self.nodes[parent].reground_seed(winner, phase);
                b += 1;
            }
            let mut b = 0usize;
            while b < 3 {
                let child = &mut self.nodes[children[b]].0;
                child.hash = seeds[b];
                child.parent = parent as u32;
                child.trits[ROOT_SENTINEL] = phase as u8;
                b += 1;
            }
            parent += 1;
        }
    }

    /// The frontier: the 81 Tier-4 nodes.
    #[inline(always)]
    pub fn frontier(&self) -> &[PertaroExpert243] {
        &self.nodes[FRONTIER_START..]
    }

    /// The frontier, mutable.
    #[inline(always)]
    pub fn frontier_mut(&mut self) -> &mut [PertaroExpert243] {
        &mut self.nodes[FRONTIER_START..]
    }
}

impl PertaroExpert243 {
    /// Consequences of the Gelfond-Lifschitz reduct of this expert's program
    /// under candidate set `s`: `Cn(P^S)`.
    ///
    /// `materialized` must be [`materialize`](Self::materialize) of this same
    /// expert; the raw word supplies the rule shape, the materialized word the
    /// rule body. Per leaf `idx` in `0..243`:
    ///
    /// - raw or materialized `Tombstone` (245): integrity constraint, never derived;
    /// - raw `NullNode` (243): fact `idx.`, always derived;
    /// - materialized value `v < 243`: rule `idx :- not v.`, derived iff `v` is not in `s`;
    /// - any other sentinel: no rule, never derived.
    pub fn gl_reduct_consequences(
        &self,
        materialized: &Word256,
        s: &AnswerSetMask,
    ) -> AnswerSetMask {
        let mut cn = AnswerSetMask::empty();
        let mut idx = 0usize;
        while idx < LEAVES {
            let raw = self.0.trits[idx];
            let v = materialized.trits[idx];
            let derived = if raw == Sentinel::Tombstone as u8 || v == Sentinel::Tombstone as u8 {
                false // integrity constraint: defeated
            } else if raw == Sentinel::NullNode as u8 {
                true // fact
            } else {
                v < MAX_PACKED && !s.contains(v) // idx :- not v.
            };
            if derived {
                cn.insert(idx as u8);
            }
            idx += 1;
        }
        cn
    }

    /// The exact stable-model test: `Cn(P^S) == S`.
    #[inline(always)]
    pub fn is_answer_set(&self, materialized: &Word256, s: &AnswerSetMask) -> bool {
        self.gl_reduct_consequences(materialized, s) == *s
    }

    /// Two-pass stratified pruning from the empty worldview: `S1 = Cn(P^∅)`,
    /// then `S2 = Cn(P^S1)`. Every coordinate leaf outside `S2` is stamped
    /// `Tombstone` in place and `S2` is returned.
    ///
    /// This is an approximation, not the fixed point: `S2` is a stable model
    /// only when [`is_answer_set`](Self::is_answer_set) says so. It is the
    /// bounded-cost pass the arena runs on every frontier node; the exact
    /// check is one call away for callers that need it.
    pub fn prune_frontier(&mut self) -> AnswerSetMask {
        let mat = self.materialize();
        let s1 = self.gl_reduct_consequences(&mat, &AnswerSetMask::empty());
        let s2 = self.gl_reduct_consequences(&mat, &s1);
        let mut idx = 0usize;
        while idx < LEAVES {
            if self.0.trits[idx] < MAX_PACKED && !s2.contains(idx as u8) {
                self.0.trits[idx] = Sentinel::Tombstone as u8;
            }
            idx += 1;
        }
        s2
    }
}

#[cfg(test)]
mod arena_tests {
    use super::*;

    fn word(seed: u64) -> Word256 {
        Word256 {
            hash: seed,
            parent: 0,
            trits: [0; PAYLOAD_BYTES],
        }
    }

    #[test]
    fn mask_insert_contains_count_and_winner() {
        let mut m = AnswerSetMask::empty();
        assert!(m.is_empty());
        assert_eq!(m.first_winner(), None);
        m.insert(200);
        m.insert(5);
        m.insert(255);
        assert!(m.contains(5) && m.contains(200) && m.contains(255));
        assert!(!m.contains(6));
        assert_eq!(m.count_ones(), 3);
        assert_eq!(m.first_winner(), Some(5));
        let mut only_sentinel = AnswerSetMask::empty();
        only_sentinel.insert(250);
        assert_eq!(only_sentinel.first_winner(), None);
    }

    #[test]
    fn arena_indexing_round_trips() {
        assert_eq!(PertaroArena121::node_index(0, 0), 0);
        assert_eq!(PertaroArena121::node_index(4, 80), ARENA_NODES - 1);
        for p in 0..FRONTIER_START {
            for c in PertaroArena121::children_of(p) {
                assert!(c < ARENA_NODES);
                assert_eq!(PertaroArena121::parent_of(c), p);
            }
        }
        assert_eq!(
            PertaroArena121::children_of(FRONTIER_START - 1)[2],
            ARENA_NODES - 1
        );
    }

    #[test]
    #[should_panic]
    fn frontier_has_no_children() {
        let _ = PertaroArena121::children_of(FRONTIER_START);
    }

    #[test]
    fn expand_is_deterministic_links_parents_and_takes_every_atom() {
        let mut a = PertaroArena121::ZERO;
        let mut b = PertaroArena121::ZERO;
        a.expand_from_root(0xDEAD_BEEF, 7, Sentinel::BusIndirect);
        b.expand_from_root(0xDEAD_BEEF, 7, Sentinel::BusIndirect);
        for i in 0..ARENA_NODES {
            assert_eq!(a.nodes[i].0, b.nodes[i].0, "node {i}");
        }
        for i in 1..ARENA_NODES {
            assert_eq!(a.nodes[i].parent() as usize, PertaroArena121::parent_of(i));
            assert_eq!(a.nodes[i].root_sentinel(), Some(Sentinel::BusIndirect));
            assert_ne!(a.nodes[i].seed(), 0);
        }
        // siblings differ
        let [c0, c1, c2] = PertaroArena121::children_of(0);
        assert_ne!(a.nodes[c0].seed(), a.nodes[c1].seed());
        assert_ne!(a.nodes[c1].seed(), a.nodes[c2].seed());
        // the paste's wrapping_add would panic here; the modulo does not
        a.expand_from_root(1, 242, Sentinel::NullNode);
        assert_eq!(a.frontier().len(), FRONTIER_NODES);
    }

    #[test]
    fn reduct_derives_facts_blocks_tombstones_and_respects_naf() {
        let mut w = word(99);
        w.trits[0] = Sentinel::NullNode as u8; // fact
        w.trits[1] = Sentinel::Tombstone as u8; // constraint
        w.trits[2] = 3; // rule 2 :- not v, with v = materialized value
        let x = PertaroExpert243(w);
        let mat = x.materialize();
        let v = mat.trits[2];
        assert!(v < MAX_PACKED);

        let cn_empty = x.gl_reduct_consequences(&mat, &AnswerSetMask::empty());
        assert!(cn_empty.contains(0));
        assert!(!cn_empty.contains(1));
        assert!(cn_empty.contains(2));

        let mut s = AnswerSetMask::empty();
        s.insert(v);
        let cn_s = x.gl_reduct_consequences(&mat, &s);
        assert!(cn_s.contains(0));
        assert!(!cn_s.contains(1));
        assert!(
            !cn_s.contains(2),
            "rule 2 :- not {v} must be deleted when {v} is in S"
        );
    }

    #[test]
    fn facts_only_program_has_exactly_one_answer_set() {
        let mut w = word(5);
        for t in w.trits[..LEAVES].iter_mut() {
            *t = Sentinel::Tombstone as u8;
        }
        w.trits[10] = Sentinel::NullNode as u8;
        w.trits[20] = Sentinel::NullNode as u8;
        let x = PertaroExpert243(w);
        let mat = x.materialize();
        let mut facts = AnswerSetMask::empty();
        facts.insert(10);
        facts.insert(20);
        assert!(x.is_answer_set(&mat, &facts));
        assert!(!x.is_answer_set(&mat, &AnswerSetMask::empty()));
        let mut too_many = facts;
        too_many.insert(30);
        assert!(!x.is_answer_set(&mat, &too_many));
    }

    #[test]
    fn prune_is_idempotent_and_leaves_only_supported_coordinates() {
        let mut x = PertaroExpert243(word(0xC0FFEE));
        for (i, t) in x.0.trits[..LEAVES].iter_mut().enumerate() {
            *t = (i % 7) as u8;
        }
        x.0.trits[0] = Sentinel::NullNode as u8;
        let s2 = x.prune_frontier();
        assert!(s2.contains(0));
        for i in 0..LEAVES {
            let raw = x.0.trits[i];
            if raw < MAX_PACKED {
                assert!(
                    s2.contains(i as u8),
                    "leaf {i} survived pruning but is not in S2"
                );
            }
        }
        let again = x.prune_frontier();
        assert_eq!(again, s2);
    }
}
