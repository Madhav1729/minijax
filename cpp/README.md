# minijax C++ backend — quick guide

Build and run tests:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
ctest --test-dir cpp/build --output-on-failure -j 4
```

Run microbench (not part of ctest):

```bash
./cpp/build/microbench
```

Notes:
- `microbench` runs two matrix sizes (N=64 and N=256) and compares baseline vs extractors.
- To add CI, run the above CMake commands and `ctest` on the build host. Keep microbench out of CI unless timing is required.
# minijax C++ backend — build & tests

Quick commands to build, test, and run benchmarks for the C++ port.

Build (out-of-source):

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build -j
```

Run unit tests (ctest):

```bash
ctest --test-dir cpp/build --output-on-failure -j 4
```

Microbenchmarks (not run in CI by default):

```bash
# small quick microbench
./cpp/build/microbench

# larger microbenchmark variant (may take longer)
./cpp/build/microbench_large
```

Notes
- Optimizer implementation and tests live in `cpp/src/opt.hpp` and `cpp/tests/*`.
- The extractors available: `extract_min_cost`, `extract_by_flops`, `extract_by_cost`.
- For CI, run the `ctest` command above on push. Microbenchmarks are local-only.
# minijax C++ port

This directory contains a C++ implementation of the same architecture described in the Rust project:

- arena-based tensor IR
- interpretable graph execution
- automatic differentiation
- optimizer rewrites
- SMT-style verification layer
- bytecode VM backend
- JIT and NN layers
- CLI and fuzzing infrastructure

The goal is to preserve the project’s conceptual structure while making it practical to build and extend in C++.

## Phase roadmap

1. Core graph + tensor engine
2. Interpreter + autodiff
3. Optimizer + rewrite passes
4. Formal verification layer
5. Bytecode VM backend
6. JIT backend
7. NN framework
8. Fuzzer and CLI
9. Benchmarking and reporting

## Build

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```
