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

/// Dependent chase: the next index comes from the word just touched, so every
/// load waits for the previous one. This exposes load-to-use latency, which
/// independent loads hide behind memory-level parallelism.
fn chase_pass(arena: &mut [PertaroExpert243], passes: usize) -> f64 {
    let n = arena.len();
    let mut i = 0usize;
    let mut acc = 0u64;
    let t = Instant::now();
    for _ in 0..passes {
        for _ in 0..n {
            let s = touch(&mut arena[i], (acc & 0xFF) as u8 % 243);
            acc ^= s;
            i = (s % n as u64) as usize;
        }
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / (passes * n) as f64
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

    // 1b. Same, with pseudo-random leaf bytes (coordinates and every sentinel
    // kind), so branch predictors see a realistic payload, not one byte repeated.
    let mut rnd = Word256 {
        hash: FNV1A64_BASIS,
        parent: 0,
        trits: [0u8; 244],
    };
    let mut g = 0x1234_5678_9ABC_DEF0u64;
    for t in rnd.trits.iter_mut() {
        g = xorshift64(g);
        *t = match g % 10 {
            0 => Sentinel::NullNode as u8,
            1 => Sentinel::Tombstone as u8,
            2 => 243 + ((g >> 8) % 13) as u8,
            _ => ((g >> 16) % 243) as u8,
        };
    }
    rnd.trits[ROOT_SENTINEL] = Sentinel::NullNode as u8;
    let rnd_expert = PertaroExpert243(rnd);
    let t0b = Instant::now();
    for _ in 0..ITERS {
        let mut e = black_box(rnd_expert);
        black_box(e.prune_frontier());
    }
    let ns_single_rnd = t0b.elapsed().as_nanos() as f64 / ITERS as f64;

    // 1c. Stage attribution on the random payload: field, materialize, one reduct pass.
    let mut field = [0u8; 243];
    let t_f = Instant::now();
    for i in 0..ITERS {
        PertaroExpert243::default_field(black_box(rnd_expert.seed() ^ i as u64), &mut field);
        black_box(&field);
    }
    let ns_field = t_f.elapsed().as_nanos() as f64 / ITERS as f64;
    let t_m = Instant::now();
    for _ in 0..ITERS {
        black_box(black_box(rnd_expert).materialize());
    }
    let ns_mat = t_m.elapsed().as_nanos() as f64 / ITERS as f64;
    let mat = rnd_expert.materialize();
    let empty = pertaro::AnswerSetMask::empty();
    let t_r = Instant::now();
    for _ in 0..ITERS {
        black_box(black_box(rnd_expert).gl_reduct_consequences(black_box(&mat), black_box(&empty)));
    }
    let ns_reduct = t_r.elapsed().as_nanos() as f64 / ITERS as f64;

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
    let sizes = [121usize, 364, 1093, 3280, 9841, 29524];
    let mut cliff = Vec::new();
    for &n in &sizes {
        let mut a = vec![expert; n];
        random_pass(&mut a, 2); // warm
        let passes = (2_000_000 / n).max(20);
        let independent = random_pass(&mut a, passes);
        let dependent = chase_pass(&mut a, passes);
        cliff.push((n, n * 256, independent, dependent));
    }

    println!("==============================================================");
    println!(
        " PERTARO LATENCY (Word256 = 256 B, Arena121 = {} B)",
        ARENA_NODES * 256
    );
    println!("==============================================================");
    println!("1. one word, materialize + reduct + prune : {ns_single:>8.1} ns  (uniform payload)");
    println!("1b. same, random payload                  : {ns_single_rnd:>8.1} ns");
    println!("1c. attribution (random payload):");
    println!("      default_field (243 digits)          : {ns_field:>8.1} ns");
    println!("      materialize (field + resolve)       : {ns_mat:>8.1} ns");
    println!("      one gl_reduct pass                  : {ns_reduct:>8.1} ns");
    println!("2. tier-4 arena, expand + 81 prunes       : {ns_l1_node:>8.1} ns/node");
    println!("3. tier-6 bank, 1093 prunes (control)     : {ns_spill_node:>8.1} ns/node");
    println!("4. light touch, by working set (tier-4 frontier starts at node {FRONTIER_START}):");
    println!("     nodes      bytes   independent ns   dependent ns");
    for (n, bytes, ind, dep) in cliff {
        println!("   {n:>7} {bytes:>10} {ind:>14.2} {dep:>14.2}");
    }
    println!("   independent = random index from an RNG (loads overlap)");
    println!("   dependent   = next index from the word just loaded (loads serialize)");
    println!("==============================================================");
}
