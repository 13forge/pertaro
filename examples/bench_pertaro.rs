//! Zero-dependency latency benchmark for PERTARO.
//!
//! Run with: `cargo run --release --example bench_pertaro`
//!
//! No cycle counter: the crate forbids `unsafe`, and `_rdtsc` needs it.
//! `Instant` is wall-clock nanoseconds; pin the process and quiet the machine
//! for stable numbers. For cache-miss counts run under
//! `perf stat -e L1-dcache-load-misses,L1-dcache-loads`.
//!
//! Sections 1 to 3 are compute-bound: `prune_frontier` is hundreds of
//! nanoseconds of arithmetic over 256 sequential bytes, so the prefetcher hides
//! the cache tier. Section 4 is the one that can show an L1 cliff: a light pass
//! over an arena of N words in pseudo-random order, so the working set, not the
//! arithmetic, sets the cost.

use pertaro::{
    xorshift64, PertaroArena121, PertaroExpert243, Sentinel, Word256, ARENA_NODES, FNV1A64_BASIS,
    FRONTIER_START, ROOT_SENTINEL,
};
use std::hint::black_box;
use std::time::Instant;

fn seeded_expert() -> PertaroExpert243 {
    let mut raw = Word256 {
        hash: FNV1A64_BASIS,
        parent: 0,
        trits: [17u8; 244],
    };
    raw.trits[0] = Sentinel::NullNode as u8;
    raw.trits[ROOT_SENTINEL] = Sentinel::NullNode as u8;
    PertaroExpert243(raw)
}

/// Light per-node work: derive a seed from the node and write it back.
/// Touches the header of one 256-byte word, roughly one cache line.
#[inline(always)]
fn touch(node: &mut PertaroExpert243, atom: u8) -> u64 {
    let s = node.reground_seed(atom, Sentinel::BusIndirect);
    node.0.hash = s;
    s
}

fn random_pass(arena: &mut [PertaroExpert243], passes: usize) -> f64 {
    let n = arena.len();
    let mut rng = FNV1A64_BASIS;
    let mut acc = 0u64;
    let t = Instant::now();
    for _ in 0..passes {
        for _ in 0..n {
            rng = xorshift64(rng);
            let i = (rng % n as u64) as usize;
            acc ^= touch(&mut arena[i], (rng >> 32) as u8 % 243);
        }
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / (passes * n) as f64
}

fn main() {
    const ITERS: usize = 100_000;
    let expert = seeded_expert();

    // 1. One word: materialize + two reduct passes + in-place tombstoning.
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let mut e = black_box(expert);
        black_box(e.prune_frontier());
    }
    let ns_single = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    // 2. Tier-4 arena: expand, prune the 81 frontier nodes, reground.
    let mut arena = Box::new(PertaroArena121 {
        nodes: [expert; ARENA_NODES],
    });
    let arena_iters = 10_000usize;
    let mut seed = FNV1A64_BASIS;
    let t1 = Instant::now();
    for step in 0..arena_iters {
        arena.expand_from_root(seed, (step % 243) as u8, Sentinel::BusIndirect);
        let mut winner = 0u8;
        for node in arena.frontier_mut() {
            if let Some(w) = node.prune_frontier().first_winner() {
                winner = w;
            }
        }
        seed = arena.nodes[ARENA_NODES - 1].reground_seed(winner, Sentinel::BusIndirect);
        black_box(seed);
    }
    let ns_l1_node = t1.elapsed().as_nanos() as f64 / (arena_iters * ARENA_NODES) as f64;

    // 3. Tier-6 flat bank, 1093 words, every node pruned: compute-bound control.
    const TIER6_NODES: usize = 1093;
    let mut bank = vec![expert; TIER6_NODES];
    let spill_iters = 2_000usize;
    let t2 = Instant::now();
    for step in 0..spill_iters {
        for (i, node) in bank.iter_mut().enumerate() {
            node.0.hash = seed.wrapping_add(i as u64);
            black_box(node.prune_frontier());
        }
        seed = seed.wrapping_add(step as u64);
    }
    let ns_spill_node = t2.elapsed().as_nanos() as f64 / (spill_iters * TIER6_NODES) as f64;

    // 4. The cliff test: light random-order touches over growing working sets.
    let sizes = [121usize, 364, 1093, 3280, 9841];
    let mut cliff = Vec::new();
    for &n in &sizes {
        let mut a = vec![expert; n];
        random_pass(&mut a, 2); // warm
        let passes = (2_000_000 / n).max(20);
        cliff.push((n, n * 256, random_pass(&mut a, passes)));
    }

    println!("==============================================================");
    println!(
        " PERTARO LATENCY (Word256 = 256 B, Arena121 = {} B)",
        ARENA_NODES * 256
    );
    println!("==============================================================");
    println!("1. one word, materialize + reduct + prune : {ns_single:>8.1} ns");
    println!("2. tier-4 arena, expand + 81 prunes       : {ns_l1_node:>8.1} ns/node");
    println!("3. tier-6 bank, 1093 prunes (control)     : {ns_spill_node:>8.1} ns/node");
    println!("4. light random touch, by working set:");
    println!("     nodes    bytes       ns/touch");
    for (n, bytes, ns) in cliff {
        let tier = match n {
            121 => "L1",
            364 | 1093 => "L2",
            _ => "L2/L3",
        };
        println!("   {n:>7} {bytes:>9} {ns:>12.2}   ({tier}, frontier start {FRONTIER_START})");
    }
    println!("==============================================================");
}
