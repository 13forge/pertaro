# `pertaro`

[![Crates.io](https://img.shields.io/crates/v/pertaro.svg)](https://crates.io/crates/pertaro)
[![Docs.rs](https://docs.rs/pertaro/badge.svg)](https://docs.rs/pertaro)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#key-guarantees)

**P**runed **E**xpert **R**egrounding **T**rie via **A**nswer-set **R**educt **O**ptimization.

`pertaro` is a `#![no_std]`, zero-dependency, zero-heap Rust primitive for cache-resident neuro-symbolic inference and bounded-horizon state planning. It packs an entire 5-tier ternary decision trie (3^5 = 243 leaves), an 8-byte deterministic PRNG seed, a 4-byte lineage link, and a 13-state in-band control envelope (`243..=255`) into an exact **256-byte cache-aligned word** (`Word256`).

---

## Core Architecture: The `243 + 13 = 256` Isomorphism

A single byte (`u8`) has 256 possible states (`0..=255`). `pertaro` partitions both the value space of a byte and the structural layout of a 256-byte cache block along the exact same boundary:

1. **Coordinate Space (`0..243`):** A 1-in, 3-out ternary trie expanded across 5 tiers yields 3^5 = **243** leaves (`MAX_PACKED = 243`). Values `0..243` represent packed 5-trit coordinates or modular residual deltas in Z_243.
2. **In-Band Control Envelope (`243..=255`):** The remaining **13** byte values (`SENTINEL_COUNT = 13`) act as in-band non-monotonic control sentinels directly inside the payload:
   - `243` (`Sentinel::NullNode`): Terminal Answer Set leaf. The deterministic `xorshift64(seed)` default prior holds at zero delta cost.
   - `244` (`Sentinel::MersenneOverflow`): Out-of-band arithmetic overflow marker.
   - `245` (`Sentinel::Tombstone`): Branch defeated by integrity constraints under the Gelfond-Lifschitz reduct.
   - `246` (`Sentinel::BusIndirect`): Horizon boundary trigger signalling a Tier-5 reground jump.
   - `247..=255` (`Sentinel::Phase5` .. `Sentinel::Phase13`): Horizon / phase anchors.

---

## Exact 256-Byte Header-First Memory Layout

Every `Word256` / `PertaroExpert243` instance is aligned to 256 bytes (`#[repr(C, align(256))]`, four 64-byte CPU cache lines) with compile-time verified field offsets:

| Byte Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| **`0..8`** | 8 B | `hash: u64` | Deterministic `xorshift64` seed and FNV-1a content signature |
| **`8..12`** | 4 B | `parent: u32` | Lineage link to the parent 5-tier horizon word |
| **`12..255`** | 243 B | `trits[0..243]` | 3^5 = 243 ternary trie leaf deltas (`0..243`) or in-band `Sentinel`s (`243..=255`) |
| **`255`** | 1 B | `trits[243]` | Active `root_sentinel` / horizon phase byte (`243..=255`) |
| **Total** | **256 B** | **`Word256`** | **128 complete 5-tier trees fit inside a single 32 KB L1 data cache** |

---

## Quick Start

This example ships as `examples/quick_start.rs` and is compiled by `cargo check --examples`.

```rust
use pertaro::{PertaroExpert243, Sentinel, Word256, MAX_PACKED, ROOT_SENTINEL};

fn main() {
    // 1. A 256-byte word with a seed, a parent link, and in-band sentinels.
    let mut raw = Word256 {
        hash: 0xcbf2_9ce4_8422_2325,
        parent: 0,
        trits: [0u8; 244],
    };

    // Address leaf [1, 0, 2, 1, 2] in the 5-tier ternary trie (each tier in 0..3).
    let target_leaf = PertaroExpert243::trie_index([1, 0, 2, 1, 2]);
    raw.trits[target_leaf] = 42; // residual override: +42 mod 243

    // Stamp a defeated branch with an in-band Tombstone (245).
    let defeated_leaf = PertaroExpert243::trie_index([2, 2, 0, 1, 0]);
    raw.trits[defeated_leaf] = Sentinel::Tombstone as u8;

    // Mark the root horizon sentinel at trits[243].
    raw.trits[ROOT_SENTINEL] = Sentinel::NullNode as u8;

    // 2. Wrap in the zero-cost view and materialize.
    let expert = PertaroExpert243(raw);
    let materialized: Word256 = expert.materialize();

    assert!(materialized.trits[target_leaf] < MAX_PACKED);
    assert_eq!(materialized.trits[defeated_leaf], Sentinel::Tombstone as u8);

    // 3. Reground the winning Tier-5 leaf (0..243) into a fresh seed.
    let next_seed = expert.reground_seed(target_leaf as u8, Sentinel::BusIndirect);
    assert_ne!(next_seed, expert.seed());
}
```

---

## Key Guarantees

* **Unbiased radix-243 extraction.** Because 243^8 = 3^40 = 12,157,665,459,056,928,801 < 2^64, `materialize()` extracts 8 independent, strictly unbiased base-243 coordinates (`0..243`) per `xorshift64` word by rejection sampling below 243^8, avoiding `u8 % 243` modulo bias entirely.
* **Strict const bounds.** `trie_index` enforces tier digits in `0..3` and `reground_seed` enforces winning leaves in `0..243`, so a sentinel is never folded as a leaf.
* **Zero unsafe, zero heap.** `#![no_std]` and `#![forbid(unsafe_code)]`, with no dynamic memory allocation anywhere.

---

## License

Licensed under either of

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
