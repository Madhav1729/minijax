# minijax-cpp

A small ML compiler written in C++20. It takes tensor programs from a text format, computes their
values and gradients, optimizes them with rewrite rules checked by the Z3
SMT solver, executes them on three backends (interpreter, bytecode VM,
LLVM ORC JIT), and trains small neural networks end to end on its own
autodiff.

116/116 tests pass. The build is warning-free under `-Wall -Wextra`.

## Build

```bash
conda env create -f environment.yml   # Eigen, Z3, llvmdev=18, toolchain
conda activate minijax-cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
./build/minijax_bench                 # benchmark numbers below
```

LLVM is optional. Without `llvm-config` on PATH the configure step skips the
JIT target (`minijax_jit`) and everything else still builds and tests.
googletest v1.15.2 is fetched by CMake FetchContent.

## Usage

```bash
./build/minijax run    examples/loss.mjx --seed 42   # eval + per-op profile
./build/minijax grad   examples/loss.mjx             # d(output)/d(every input)
./build/minijax opt    examples/loss.mjx             # optimize_sound stats
./build/minijax opt    examples/loss.mjx --fast-math # allow float-unsafe rules
./build/minijax verify                               # Z3 soundness report
./build/minijax fuzz   --iters 100 --seed 42 --jit   # 3-oracle campaign
./build/minijax train  --dataset two-moons           # train an MLP to ~100%
./build/minijax viz    examples/loss.mjx --grad -o g.dot && dot -Tpng g.dot -o g.png
./build/minijax memplan examples/mlp.mjx             # memory plan report
./build/minijax repl                                 # incremental .mjx session
```

Example program:

```
let W = input([2, 2]);
let x = input([2, 1]);
let y = input([2, 1]);
let pred = relu(W @ x);
output sum(pred - y);
```

Operators: `+ - * / @` (matmul), unary `-`, postfix `'` (transpose).
Functions: `input([d,...])`, `relu`, `step`, `tanh`, `sigmoid`, `exp`,
`log`, `sqrt`, `abs`, `sum`, `sum_axis(x,k)`, `transpose`,
`reshape(x,[d,...])`, `broadcast(x,[d,...])`. Comments start with `#`.

## Modules

| Module | Files | Purpose |
|---|---|---|
| Tensor | src/tensor.cpp | dynamic-rank row-major tensor; Eigen Map GEMM inside matmul only |
| IR | src/ir.cpp | arena graph in topological order; eager shape inference; explicit Broadcast nodes |
| Interpreter | src/interp.cpp | single-pass evaluator; reference oracle for all other backends; optional per-op profiler |
| Autodiff | src/autodiff.cpp | reverse mode appending gradient nodes into the same graph; numeric grad checks at 1e-4 |
| Optimizer | src/egraph.cpp, src/opt.cpp | equality saturation (commutativity, associativity, identities, matmul re-assoc); cost-based extraction |
| Verifier | src/verify.cpp | checks each rule over reals and IEEE-754 doubles; drives optimize_sound |
| VM | src/vm.cpp | register machine, last-use buffer reuse, binary serialization |
| JIT | src/jit.cpp | ORCv2 LLJIT; emitted loops for elementwise ops; imported shared matmul kernel |
| Frontend | src/frontend/ | lexer, recursive-descent parser, lowering; errors carry line numbers |
| Memory planner | src/memplan.cpp | shared liveness analysis, numel-bucketed slot allocator, peak-memory report |
| NN | src/nn.cpp | MLP layers, SGD/Adam, two-moons dataset, training loop |
| Fuzzer | src/fuzz.cpp | random program generator, three oracles, delta-debug minimizer |
| CLI / viz | src/cli.cpp, src/viz.cpp | subcommands over every subsystem; DOT export |

## Benchmarks

Measured with `./build/minijax_bench 50` (Release -O2, conda-forge GCC,
x86-64). Ratios matter more than absolute times.

| Workload | interpreter | VM | JIT |
|---|---|---|---|
| matmul chain 32x32, depth 30 | 0.280 ms | 0.157 ms | 0.161 ms |
| elementwise n=4096, depth 64 | 3.747 ms | 2.091 ms | 2.085 ms |

The matmul case is dominated by the shared Eigen kernel that every backend
calls, so backend speedup is bounded by dispatch overhead. The elementwise
case is memory-bandwidth bound at these sizes.

MLP{2,16,16,2} forward+backward+Adam step: about 6 ms per 240-sample epoch
(~40k samples/s); reaches 100% accuracy on two-moons within 60 epochs.

## Floating-point soundness

The verifier checks each rewrite rule over ideal reals and again over
IEEE-754 doubles.

- `(a+b)+c != a+(b+c)` has genuine finite-value rounding counterexamples;
  associativity is not float-safe.
- Commutativity, add-zero, mul-one, and double negation fail float checking
  only through NaN semantics (`fp.eq(NaN,NaN)` is false). Counterexamples
  are annotated as NaN-driven rather than treated as real unsoundness.
- `0 * inf = NaN != 0`, so mul-zero is genuinely float-unsafe.

`optimize_sound()` excludes associativity and mul-zero; `opt --fast-math`
allows them. A test injects a deliberately wrong rule and confirms the
verifier rejects it over the reals as well. Counterexample shapes differ
across Z3 versions (4.8 picks NaN operands, 5.x prefers `-0 * +oo`), so the
classifier recognizes both shapes and tests assert behavior, not spellings.

## Writing tests

Tests use GoogleTest, one file per module under `tests/`. CMake globs
`tests/*.cpp`; re-run `cmake -B build` after adding a file.

```cpp
#include <gtest/gtest.h>
#include "minijax/frontend.hpp"
#include "minijax/interp.hpp"
#include "fixtures.hpp"

using namespace minijax;

TEST(MyFeature, Description) {
    auto f = test_fixtures::make_loss_fixture();
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    Tensor want = eval(f.g, inputs, f.loss);
    // exercise your feature, then:
    // EXPECT_TRUE(Tensor::allclose(want, got, 1e-9, 1e-9));
}
```

Conventions:

- Backends must match `interp::eval` within 1e-9 (see jit_test.cpp,
  vm_test.cpp).
- New autodiff rules get a central-difference check via
  `tests/gradcheck.hpp` (tol 1e-4).
- Shape errors throw `std::invalid_argument`; assert on message content when
  readability is the feature.

```bash
ctest --test-dir build                                  # all tests (~20s)
./build/minijax_tests --gtest_filter='Jit.*'            # one suite
./build/minijax_tests --gtest_filter='Vm.MatchesInterpreterOnLossFixture'
```

## Layout

```
include/minijax/   public headers, one per module
src/               implementations (+ src/frontend/)
tests/             GoogleTest files, shared fixtures and gradcheck helper
bench/bench.cpp    timing harness
examples/*.mjx     sample programs
IMPLEMENTATION_PLAN.md   phase log, invariants, bugs found
environment.yml    conda environment
```
