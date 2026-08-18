# minijax

This repository contains the Rust implementation and a C++ port focused on the IR, interpreter, autodiff, optimizer, and runtime validation flow.

## Quick start

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure -j 4
```

## Benchmarking

The C++ port includes lightweight microbenchmarks for optimizer extraction strategies.

```bash
cmake --build cpp/build --target microbench -j
./cpp/build/microbench

cmake --build cpp/build --target microbench_large -j
./cpp/build/microbench_large
```

These compare the baseline graph execution against optimized forms using the rewrite/extractor pipeline.

## Project status

The C++ port is implemented and validated for:

- IR and interpreter
- reverse-mode autodiff
- optimizer passes for const folding and associative/commutative rewrites
- fixed-point optimization
- cost-based extraction heuristics
- numeric and optimizer regression tests

The remaining items are future work: backend portability, wider op coverage, VM/JIT work, and full project packaging.
