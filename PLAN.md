# minijax — Build Plan

A differentiable array-programming compiler in Rust where **every optimization is
formally proven sound by an SMT solver** — including the discovery that several
kernel-fusion rewrites are valid over real arithmetic but *unsound* under IEEE-754
floating point.

Full stack: language frontend → autodiff → equality-saturation optimizer (egg) →
SMT verification (Z3) → multiple backends (interpreter / bytecode VM / Cranelift JIT)
→ NN framework → differential fuzzer.

---

## What this is (and isn't)

- **Not** novel research. JAX/PyTorch/XLA exist; this reimplements their core.
  Its value is a **capability signal** — it proves deep understanding across
  autodiff + optimization + codegen + formal methods, which is hard to fake.
- **The one research-flavored lever:** verify each rewrite rule twice — over Z3's
  `Real` theory and its IEEE-754 `Float` theory — and classify rules into
  {sound everywhere / sound over reals only / unsound}. Float arithmetic is not
  associative, so fusions like matmul-reassociation are technically unsound under
  IEEE-754 (production compilers gate these behind `-ffast-math`). This is a true,
  measured finding, and the verifier will also catch real bugs in your own rules.
- Prior art to know (so you're not blindsided): TASO verifies tensor-graph rewrites
  with Z3; Alive2 verifies LLVM peephole opts. The novel part here is the *specific
  combination* + the reals-vs-IEEE-754 soundness classification.

## Resume line

> Built a differentiable array-programming compiler in Rust (reverse-mode autodiff,
> egg-based kernel fusion, Cranelift JIT) with an SMT verification layer (Z3) that
> formally proves every optimization sound — discovering that several fusion rewrites
> are valid only over real arithmetic, not IEEE-754, and catching incorrect rules
> pre-deployment.

---

## Architecture & LOC budget (~13,000–18,000 lines)

| Component                  | Lines        |
|----------------------------|--------------|
| Text frontend              | 1,200–1,600  |
| IR + core                  | 400–600      |
| Expanded op set (~30 ops)  | 1,800–2,500  |
| Autodiff                   | 700–1,000    |
| Optimizer                  | 1,200–1,700  |
| Verification layer         | 1,200–1,700  |
| Bytecode VM backend        | 800–1,200    |
| Cranelift JIT backend      | 600–900      |
| Second codegen backend     | 800–1,200    |
| NN framework               | 1,000–1,500  |
| Differential fuzzer        | 1,000–1,500  |
| Tooling                    | 800–1,300    |
| Tests / benches            | 1,200–1,800  |

The crates (egg, cranelift, z3) do the giant lifting; you write ~13K lines that
orchestrate ~100K+ lines of dependencies. Normal and expected.

**Hard tasks — understand every line (interview-defensible core):**
2.5, 3.2/3.3, 4.5, 6.3, 7.2, 9.1, 12.2. Don't let AI black-box these.

**Stop-anywhere safety:** the project is real and demoable after Phase 8.
Phases 9–13 push it past 10K lines and into "wow."

**Biggest LOC multipliers** (if you need to be sure of clearing 10K): Phases 7, 11, 12.

---

## Phase 0 — Setup
- [ ] 0.1 `cargo new minijax`, set edition 2021
- [ ] 0.2 Add deps: `egg`, `ndarray`, `rand`, `z3`, `cranelift`/`cranelift-jit`/`cranelift-module`, `approx`, `proptest` (dev)
- [ ] 0.3 Create module skeleton: `ir`, `interp`, `autodiff`, `opt`, `verify`, `vm`, `jit`, `nn`, `fuzz`, `viz`, `cli`
- [ ] 0.4 Set up `cargo test` + a `make check` (fmt + clippy + test)

## Phase 1 — IR + interpreter (vertical slice, 5 ops)
- [ ] 1.1 Define `Op` enum: `Input, Const, Add, Mul, MatMul, Relu, Sum`
- [ ] 1.2 Define `Node` (op + input ids + shape) and arena `Graph`
- [ ] 1.3 Builder methods (`add`, `mul`, `matmul`, `relu`, `sum`, `input`, `constant`) with shape computation
- [ ] 1.4 Shape inference + validation (reject shape-mismatched matmul/add)
- [ ] 1.5 Interpreter: topological eval over `ndarray::ArrayD<f64>`
- [ ] 1.6 The 5 tensor kernels (matmul, broadcasted add/mul, relu, sum)
- [ ] 1.7 Build `loss(W,x,y)` by hand; unit-test against hand-computed values
- [ ] ✅ **Gate: interpreter runs programs and prints correct results**

## Phase 2 — Autodiff + gradient check
- [ ] 2.1 `grad(graph, output, wrt)` skeleton: adjoint map + reverse traversal
- [ ] 2.2 `accumulate` helper (build Add node when adjoint already exists)
- [ ] 2.3 Backward rules for Add, Mul, Sum
- [ ] 2.4 Add `Step`, `Transpose`, `Broadcast` ops to IR + interp (needed by relu/matmul/sum grads)
- [ ] 2.5 Backward rules for Relu, MatMul  *(HARD)*
- [ ] 2.6 Numerical gradient-check harness (central differences, `assert_relative_eq!`)
- [ ] 2.7 Per-op gradient tests, then full-`loss` gradient test
- [ ] 2.8 Forward-mode AD (JVP) as a cross-check + warm-up
- [ ] ✅ **Gate: gradient check passes @1e-5 for every op and the full loss**

## Phase 3 — Equality-saturation optimizer
- [ ] 3.1 `define_language!` egg enum mirroring the IR ops
- [ ] 3.2 `Graph → RecExpr` serialization  *(HARD)*
- [ ] 3.3 `RecExpr → Graph` deserialization  *(HARD)*
- [ ] 3.4 Algebraic rewrite rules (add-zero, mul-one, mul-zero, comm)
- [ ] 3.5 Structural rules (matmul-assoc, sum-fusion)
- [ ] 3.6 `CostFunction` (FLOP/kernel-count cost model)
- [ ] 3.7 Runner + Extractor → `optimize(graph) -> graph`
- [ ] 3.8 Op-count + FLOP measurement before/after
- [ ] 3.9 Correctness test: `eval(opt) == eval(orig)` numerically
- [ ] ✅ **Gate: optimizer cuts N% of ops, outputs bit-identical**

## Phase 4 — SMT verification layer (the differentiator)
- [ ] 4.1 Z3 encoding of scalar ops over `Real` theory
- [ ] 4.2 Verify scalar/elementwise rules valid (assert-negation-unsat)
- [ ] 4.3 Z3 encoding over `Float` (IEEE-754) theory
- [ ] 4.4 Re-verify all rules over floats; capture SAT counterexamples
- [ ] 4.5 Bounded tensor-rule verification (unroll sums for fixed dims N=1,2,3)  *(HARD)*
- [ ] 4.6 Rule classification: {sound everywhere / reals-only / unsound}
- [ ] 4.7 Rule DSL: each rewrite carries its proof obligation; verifier gates it into the rule set
- [ ] 4.8 Soundness report generator (table + printed float counterexamples)
- [ ] 4.9 Deliberately add a wrong rule; confirm verifier catches it
- [ ] ✅ **Gate: rule classification table produced; fast-math rules flagged; verifier catches a bad rule**

## Phase 5 — Bytecode VM backend
- [ ] 5.1 Design instruction set (load/store/binop/matmul/reduce)
- [ ] 5.2 Graph → bytecode compiler (linearize, assign virtual registers)
- [ ] 5.3 Register/buffer allocation (reuse dead buffers)
- [ ] 5.4 VM executor over the bytecode
- [ ] 5.5 Bytecode (de)serialization to disk
- [ ] 5.6 Differential test: VM output == interpreter output
- [ ] ✅ **Gate: second backend runs, matches interpreter**

## Phase 6 — Cranelift JIT backend
- [ ] 6.1 Cranelift setup (JITModule, function signatures)
- [ ] 6.2 Lower elementwise ops to generated loops
- [ ] 6.3 matmul via imported hand-written runtime kernel  *(HARD)*
- [ ] 6.4 Lower reductions + remaining ops
- [ ] 6.5 Callable fn pointer over data buffers
- [ ] 6.6 Differential test: JIT == interpreter == VM
- [ ] 6.7 Benchmark JIT vs interpreter
- [ ] ✅ **Gate: JIT matches other backends, Nx speedup measured**

## Phase 7 — Widen the op set
- [ ] 7.1 softmax + cross-entropy (forward/backward/interp/jit/verify)
- [ ] 7.2 conv2d + backward (the big one)  *(HARD)*
- [ ] 7.3 max-pool + backward
- [ ] 7.4 batchnorm + backward
- [ ] 7.5 tanh, sigmoid, more elementwise
- [ ] 7.6 Extend optimizer rules + cost model for new ops
- [ ] 7.7 Gradient-check all new ops
- [ ] ✅ **Gate: ~30 ops, all gradient-checked, all backends support them**

## Phase 8 — NN framework + training
- [ ] 8.1 Layer abstractions (Linear, Conv, BN, activations)
- [ ] 8.2 Parameter management + init
- [ ] 8.3 Optimizers: SGD, then Adam
- [ ] 8.4 Data loader + batching (synthetic + MNIST loader)
- [ ] 8.5 Build + train an MLP on two-moons/MNIST subset
- [ ] 8.6 Build + train a small CNN on MNIST
- [ ] 8.7 Log loss/accuracy curves
- [ ] ✅ **Gate: a real net trains end-to-end on your stack**

## Phase 9 — Differential fuzzer
- [ ] 9.1 Random valid-program generator (type/shape-correct, no UB)  *(HARD)*
- [ ] 9.2 Cross-backend oracle: interp vs VM vs JIT must agree
- [ ] 9.3 Metamorphic oracle: optimized vs unoptimized must agree
- [ ] 9.4 Failure minimizer (delta-debug the program)
- [ ] 9.5 Dedup + triage of failures
- [ ] 9.6 Run campaign, log/fix bugs found in your own optimizer
- [ ] ✅ **Gate: fuzzer runs unattended, finds (and you fix) real bugs**

## Phase 10 — Text frontend (language polish)
- [ ] 10.1 Lexer (`logos`)
- [ ] 10.2 Recursive-descent / `chumsky` parser → AST
- [ ] 10.3 AST → IR lowering
- [ ] 10.4 Type checker with shape inference + good error messages
- [ ] 10.5 `.mjx` source files for the example programs
- [ ] ✅ **Gate: programs run from source text, not just the Rust API**

## Phase 11 — Second codegen backend (pick one)
- [ ] 11.1 Choose: C-emit *or* LLVM (`inkwell`) *or* GPU (`wgpu`)
- [ ] 11.2 Graph → target lowering for core ops
- [ ] 11.3 Build/link/run pipeline
- [ ] 11.4 Differential test against existing backends + benchmark
- [ ] ✅ **Gate: third backend validated, perf compared**

## Phase 12 — Advanced optimizer passes
- [ ] 12.1 Liveness analysis over the linearized graph
- [ ] 12.2 Memory planner (buffer reuse / in-place ops)  *(HARD)*
- [ ] 12.3 Operator-fusion scheduling pass
- [ ] 12.4 Layout selection (row/col-major) via cost model
- [ ] 12.5 Measure memory + speed wins
- [ ] ✅ **Gate: measurable memory-traffic reduction**

## Phase 13 — Tooling
- [ ] 13.1 Graphviz/DOT export of forward + gradient graphs
- [ ] 13.2 CLI subcommands (`run`, `grad`, `opt`, `verify`, `fuzz`, `train`, `viz`)
- [ ] 13.3 Simple REPL
- [ ] 13.4 Profiler (per-op timing)
- [ ] ✅ **Gate: usable CLI + visual graph output**

## Phase 14 — Polish + write-up
- [ ] 14.1 Property tests (`proptest`) for AD and optimizer invariants
- [ ] 14.2 Benchmark harness + results table
- [ ] 14.3 README: architecture diagram, the metrics, the float-soundness finding
- [ ] 14.4 Short design doc (AD/egg/Z3/fuzzer decisions + tradeoffs)
- [ ] ✅ **Gate: clean repo, headline numbers, design writeup**
