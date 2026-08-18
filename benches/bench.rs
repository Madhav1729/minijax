// Benchmark harness — Phase 14.
// Will benchmark interpreter vs VM vs JIT and optimizer before/after.
use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder_bench(_c: &mut Criterion) {
    // TODO Phase 14
}

criterion_group!(benches, placeholder_bench);
criterion_main!(benches);
