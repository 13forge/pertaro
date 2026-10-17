//! The README's Quick Start, kept compiling by `cargo check --examples`.

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
    assert_eq!(materialized.hash, expert.seed());

    // 3. Reground the winning Tier-5 leaf (0..243) into a fresh seed.
    let next_seed = expert.reground_seed(target_leaf as u8, Sentinel::BusIndirect);
    assert_ne!(next_seed, expert.seed());

    println!("leaf {target_leaf} -> {}", materialized.trits[target_leaf]);
    println!("next seed {next_seed:#018x}");
}
