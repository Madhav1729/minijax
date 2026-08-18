# minijax — Detailed Implementation Plan

This document is a **build-ready** companion to `PLAN.md`. It records the current
state, the codebase conventions you must not break, the library API gotchas
already discovered (z3 0.20, egg 0.9, cranelift 0.115, ndarray 0.16), and a
concrete, step-by-step plan for every remaining phase — with data structures,
algorithms, edge cases, and per-file tests. It is written so an implementer can
proceed one phase at a time without rediscovering the tricky parts.

Golden rule: **after each numbered step, run `cargo test <module>::` and keep it
green before moving on.** Never advance a phase with a failing test.

---

## 0. Current state (as of this plan)

| Phase | Module(s)                | Status | Tests |
|-------|--------------------------|--------|-------|
| 0 Setup            | `Cargo.toml`, `justfile` | ✅ done | — |
| 1 IR + interp      | `src/ir.rs`, `src/interp.rs` | ✅ done | 11 |
| 2 Autodiff         | `src/autodiff.rs`        | ✅ done | 10 (grad-check + fwd/rev) |
| 3 Optimizer (egg)  | `src/opt.rs`             | ✅ done | 3 |
| 4 SMT verify (Z3)  | `src/verify.rs`          | ⚠️ implemented, confirm tests pass | 5 (written) |
| 5 VM backend       | `src/vm.rs`              | ⬜ stub | — |
| 6 Cranelift JIT    | `src/jit.rs`             | ⬜ stub | — |
| 7 Widen op set     | across modules           | ⬜ stubs (`unimplemented!`) | — |
| 8 NN framework     | `src/nn.rs`              | ⬜ stub | — |
| 9 Fuzzer           | `src/fuzz.rs`            | ⬜ stub | — |
| 10 Text frontend   | `src/frontend/` (new)    | ⬜ not started | — |
| 11 C-emit backend  | `src/cbackend.rs` (new)  | ⬜ not started | — |
| 12 Adv. optimizer  | `src/memplan.rs` (new)   | ⬜ not started | — |
| 13 Tooling/CLI     | `src/cli.rs`, `src/viz.rs` | ⬜ stubs | — |
| 14 Polish/benches  | `benches/bench.rs`, `README.md` | ⬜ stub | — |

**Phase 4 note:** the Z3 build (bundled) is slow on a cold cache (~minutes). The
tests are written and expected to pass. If `verify::tests` fail, check the two
likely spots flagged in §4 below (the `_eq`/`!` operator on `Bool`, and
`add_with_rounding_mode` argument-by-value vs by-ref). No logic redesign needed.

---

## 1. Codebase conventions & invariants (DO NOT BREAK)

These hold across all phases. New code must respect them.

1. **Graph is an arena, topologically ordered by construction.** `NodeId = usize`
   is an index into `Graph::nodes`. A node's `inputs` only reference *earlier*
   indices. Therefore iterating `0..nodes.len()` is a valid forward topo order,
   and `(0..n).rev()` is a valid reverse topo order. Preserve this — never insert
   a node that references a later index.

2. **No implicit broadcasting inside ops.** Binary ops require operands of
   *equal* shape; the builder (`Graph::binop`) inserts explicit `Broadcast`
   nodes. Any new binary op must do the same (call `self.broadcast_to`).

3. **`Op::Input(i)` carries its runtime slot index `i`.** `eval(g, inputs)`
   reads `inputs[i]`. `Graph::inputs: Vec<NodeId>` maps slot → node id, and
   `num_inputs()` is its length. When rebuilding graphs (optimizer, fuzzer),
   preserve input arity and indices (see `opt::from_recexpr`, which pre-sizes
   `g.inputs` with `usize::MAX` placeholders for dropped inputs).

4. **Tensors are `ndarray::ArrayD<f64>`** (`interp::Tensor`). Scalars are
   0-dimensional (`IxDyn(&[])`), NOT shape `[1]`. `t.first().unwrap()` reads a
   scalar; `ArrayD::from_elem(IxDyn(&[]), v)` builds one.

5. **Every backend must be differentially tested against the interpreter.**
   `interp::eval` is the reference oracle. VM, JIT, C-emit outputs must match it
   within `1e-9` (they do the same f64 ops in the same order, so usually exact).

6. **`matmul_op` is `pub(crate)` in interp** and is the shared matmul kernel —
   reuse it (forward-mode AD already does).

7. **Gradient correctness is checked numerically.** The pattern for any new
   differentiable op: add a `grad_<op>` test in `autodiff.rs` using the existing
   `check(build, inputs)` harness (central differences, tol 1e-4). If it passes,
   the backward rule is correct.

8. **egg `Mjx` language mirrors `Op`.** Any op you want the optimizer to see must
   get a `Mjx` variant + arms in `to_recexpr`, `from_recexpr`, `ShapeAnalysis`,
   and `FlopCost`. Ops NOT added there will hit the `panic!("op not supported by
   optimizer yet")` — so the optimizer currently rejects Softmax/Conv/etc. That
   is fine; extend it in Phase 7.6 only for the ops you want fused.

---

## 2. Library API gotchas (already discovered — trust these)

### ndarray 0.16
- `ArrayD::from_shape_vec(IxDyn(&shape), vec)` — build from flat data (row-major).
- `.mapv(|x| ...)` elementwise; `.sum()`, `.sum_axis(Axis(i))`.
- `.broadcast(IxDyn(&shape))` returns `Option<view>`; scalars (ndim 0) don't
  broadcast — special-case with `from_elem` (see `interp::Op::Broadcast`).
- `.into_shape_with_order(IxDyn(&shape))` for reshape.
- `.t()` transposes (2-D), `.to_owned()` to materialize.
- Element access: `a[IxDyn(&[i, j])]`. Flat iteration: `.iter()` (row-major).

### egg 0.9.5
- `define_language!` supports `Box<[Id]>` variadic children (used for `Shape`).
- **`Analysis::make(egraph: &EGraph, enode: &L)` takes `&EGraph` (immutable).**
  (Not `&mut` — that was a compile error we already fixed.)
- Need `use egg::Language;` in scope for `enode.fold(...)` and `.children()`.
- **Root id gotcha:** `to_recexpr` returns a *RecExpr-local* `Id`. After
  `Runner::with_expr(&expr).run(...)`, get the e-graph root from
  `runner.egraph.find(runner.roots[0])`, NOT from the RecExpr id (they differ
  because the e-graph dedups). This bit us once — index-out-of-bounds in unionfind.
- Custom `Applier`/condition: closures `Fn(&mut EGraph, Id, &Subst) -> bool` work
  as rewrite conditions; `ConstApplier` in `opt.rs` shows a hand-rolled applier
  that inspects an e-class for a `Const` node of a given value.
- `Extractor::new(&egraph, cost_fn)` + `.find_best(root) -> (Cost, RecExpr)`. Both
  the extractor and a cost fn holding `&egraph` borrow immutably — that's fine.

### z3 0.20.0 (thread-local context; NO explicit `Context`/`ctx` args)
- Consts: `Real::fresh_const("r")`, `Float::fresh_const_double("f")`. Uniqueness
  is automatic (fresh). Use `new_const(name)` only if you want a named symbol.
- `Real` supports `+ - *` operators and unary `-` (via `&r`). `_eq(&other)`
  returns `Bool`. Build integer constants with `Real::from_real(num, den)`.
- **`Float` has NO `+ - *` operators** (rounding is explicit):
  `a.add_with_rounding_mode(b, &rm)`, `sub_with_rounding_mode`,
  `mul_with_rounding_mode`. Args are `IntoAst<Float>` — pass **owned** `Float`
  (clone if needed). Unary neg: `-(&f)`. Constant: `Float::from_f64(x)`.
- Rounding mode: `RoundingMode::round_nearest_ties_to_even()`.
- `Bool` negation: `!bool_ast` (the `Not` operator is implemented) — used as
  `solver.assert(&!lhs._eq(&rhs))`.
- `Float::_eq` is Z3 **structural** equality (bit pattern; one NaN class; +0 ≠ −0).
  To restrict to finite values use the identity `(a - a) == +0` — true iff `a` is
  neither NaN nor ±∞ (see `verify::floats_sound`). This is the whole trick that
  keeps the float encoding FFI-free.
- Solver: `Solver::new()`, `.assert(&bool)`, `.check() -> SatResult` (`Sat`,
  `Unsat`, `Unknown`), `.get_model() -> Option<Model>`.
- Model read-back: `model.eval(&float_const, true) -> Option<Float>`, then
  `.as_f64()`.

### cranelift 0.115 (for Phase 6 — verify against real API when you get there)
- Crates already in `Cargo.toml`: `cranelift-codegen`, `-frontend`, `-jit`,
  `-module`, `-native`.
- Typical flow: `JITBuilder::new(cranelift_native::builder()?.finish(...))` →
  `JITModule::new(builder)`. Build a `Function` with `FunctionBuilder`, define
  signature, emit CLIF, `module.define_function`, `module.finalize_definitions`,
  then `module.get_finalized_function(func_id)` → transmute to an `extern "C" fn`.
- See §6 for the exact plan; the matmul kernel is imported as a symbol, not
  emitted as CLIF.

---

## 3. Phase 4 — finish/confirm (SMT verify)

Already implemented in `src/verify.rs`. To close it out:

1. `cargo test verify::` — expect 5 passing tests:
   `identities_sound_everywhere`, `reassociation_is_reals_only`,
   `signed_zero_breaks_add_zero_mul_zero`, `verifier_catches_a_bad_rule`,
   `matmul_assoc_k1_sound_k2_fastmath`.
2. If a compile error appears, it will be in one of:
   - `!lhs._eq(&rhs)` — if `Not` isn't resolving, use `lhs._eq(&rhs).not()` with
     `use std::ops::Not;`.
   - `add_with_rounding_mode(to_float(...), rm)` — if the arg type is rejected,
     the value is already owned `Float`; ensure you're not passing `&Float`.
3. Wire the report into the CLI in Phase 13 (`minijax verify` prints
   `verify::soundness_report()`).
4. **Connect to the optimizer (Phase 4.7 "gating"):** add to `opt.rs` a
   `pub fn optimize_sound(g, out)` that filters `rules()` to exclude any rewrite
   whose name is in `verify::fast_math_rule_names()` (i.e. exclude `add-zero`,
   `mul-zero`, the assoc rules). Keep the existing `optimize` as the fast-math
   (all-rules) variant. Add a test that `optimize_sound` leaves an
   `(a+b)+c` association unchanged while `optimize` may reassociate. This is the
   payoff that ties Phases 3+4 together and should be done before Phase 5.

Done-criteria: soundness table prints; fast-math rules flagged; bad rule caught;
`optimize_sound` provably avoids reassociation.

---

## 4. Phase 5 — Bytecode VM backend (`src/vm.rs`)

**Goal:** a second execution engine that linearizes the graph into a flat
instruction stream over a register/buffer file, runs it, and matches the
interpreter bit-for-bit. Also serialize/deserialize the bytecode.

### 5.1 Instruction set
Define registers as `type Reg = usize` (index into a `Vec<Tensor>` slot file).
```rust
pub enum Instr {
    LoadInput { dst: Reg, idx: usize },
    Const     { dst: Reg, val: f64, shape: Vec<usize> },
    Unary     { dst: Reg, op: UnOp, a: Reg },          // Neg,Relu,Step,Exp,Log,Tanh,Sigmoid
    Binary    { dst: Reg, op: BinOp, a: Reg, b: Reg }, // Add,Sub,Mul,Div
    MatMul    { dst: Reg, a: Reg, b: Reg },
    Transpose { dst: Reg, a: Reg },
    Broadcast { dst: Reg, a: Reg, shape: Vec<usize> },
    Reshape   { dst: Reg, a: Reg, shape: Vec<usize> },
    Sum       { dst: Reg, a: Reg },
    SumAxis   { dst: Reg, a: Reg, axis: usize },
}
pub struct Program { pub instrs: Vec<Instr>, pub num_regs: usize, pub output: Reg, pub num_inputs: usize }
```
`UnOp`/`BinOp` are small enums mirroring the elementwise `Op`s.

### 5.2 Compile Graph → Program
- Walk `g.nodes` in order (already topo). Maintain `reg_of: Vec<Reg>` (node → reg).
- Naïve allocation first: **one register per node** (`dst = node_id`,
  `num_regs = nodes.len()`). This is trivially correct; get it passing before
  optimizing allocation.
- Map each `Op` to an `Instr`. `output = reg_of[output_node]`.

### 5.3 Buffer/register reuse (5.3)
- Compute **last-use** for each node: `last_use[n] = max index of any consumer`
  (or the output). Single reverse pass.
- Allocate with a free-list: iterate instructions in order; when emitting an
  instr whose result reg can reuse a buffer whose last-use has passed, pop from
  the free list; after emitting, push any input regs whose last-use == current
  index back to the free list. This reduces `num_regs` well below `nodes.len()`.
- Keep the naïve path behind a flag or as the fallback; test both produce equal
  output. **Gotcha:** never free a buffer that is also the output reg, and never
  reuse a reg that is still live (aliasing bug — this is exactly what the fuzzer
  in Phase 9 will hunt for, so get it right).

### 5.4 Executor
- `pub fn run(p: &Program, inputs: &[Tensor]) -> Tensor`.
- `let mut regs: Vec<Option<Tensor>> = vec![None; p.num_regs];`
- Match each instr, reusing the exact same ndarray kernels as `interp` (factor
  the elementwise/matmul logic into small helpers shared by interp and vm, or
  just replicate — replication is fine and keeps modules decoupled).
- Return `regs[p.output].take().unwrap()`.

### 5.5 Serialization
- Derive `Serialize`/`Deserialize`? `serde` is NOT in deps. Simplest: hand-write
  a compact binary or text format, OR add `serde` + `bincode`. **Recommendation:**
  avoid new deps — implement `Program::to_bytes(&self) -> Vec<u8>` and
  `from_bytes(&[u8]) -> Program` with a trivial manual encoding (tag byte per
  instr + little-endian fields). Round-trip test: `from_bytes(to_bytes(p))`
  executes identically.

### 5.6 Tests (`vm::tests`)
- `vm_matches_interp_elementwise`, `..._matmul`, `..._full_loss` — build the same
  graphs used in interp/autodiff tests, run both, assert equal (use `==` on
  ArrayD since ops are identical order → bit-exact).
- `buffer_reuse_shrinks_regs` — assert `num_regs < nodes.len()` on a chain graph
  and output still matches.
- `bytecode_roundtrips` — serialize/deserialize equality.

Done-criteria: VM output == interpreter on all shared test graphs; reuse shrinks
register count; bytecode round-trips.

---

## 5. Phase 6 — Cranelift JIT backend (`src/jit.rs`)

**Goal:** JIT-compile a graph to native code operating on raw `*mut f64`
buffers; match interpreter; measure speedup. This is the HARD one (6.3).

### Strategy (keep it tractable)
Do **not** try to fuse or vectorize. Compile the graph to a single Cranelift
function that, for each node in topo order, calls either:
- an **emitted scalar loop** (elementwise ops, broadcast, reshape, reductions), or
- an **imported runtime kernel** (matmul, and optionally transpose) — a plain
  Rust `extern "C"` function linked as a symbol.

Buffers: allocate one `Vec<f64>` per node (reuse Phase 5's last-use analysis
later if time permits). The JIT function receives a pointer to a table of buffer
pointers, or simpler: generate a function `fn(inputs: *const *const f64, out: *mut f64)`
and have the codegen bake buffer offsets. **Simplest workable design:**

1. Host side allocates a `Vec<Vec<f64>>` (one flat buffer per node, sized from
   shapes). Pass a `*mut *mut f64` (array of buffer base pointers) + `*const *const f64`
   inputs into the JIT'd function.
2. The JIT function body, per node, emits code that reads from child buffer
   pointers and writes the node's buffer, using known static shapes (loop bounds
   are constants baked at compile time).

### 6.1 Setup
```rust
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, Linkage, FuncId};
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, ...};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
```
- `let mut builder = JITBuilder::new(cranelift_native::builder().unwrap().finish(settings::Flags::new(settings::builder())).unwrap()).unwrap();`
- Register imported kernels: `builder.symbol("mj_matmul", mj_matmul as *const u8);`
- `let mut module = JITModule::new(builder);`

### 6.2 Emit elementwise as loops
- Function signature: params are pointers (`types::I64` holding addresses) + an
  `i64` element count where needed. Use `pointer_type = module.target_config().pointer_type()`.
- For an elementwise unary over N elements: emit a loop block with an index var
  (use `Variable`s or block params), `load.f64` from `a_ptr + i*8`, apply op
  (`fadd`, `fmul`, `fsub`, `fdiv`, `fneg`; relu = `fmax` with 0; exp/log/tanh/
  sigmoid → call libm via imported symbols `mj_exp`, `mj_tanh`, etc.), `store` to
  `dst_ptr + i*8`, increment, branch while `i < N`.
- Broadcast/reshape: emit a copy loop with index mapping (reshape is a straight
  copy; broadcast repeats — compute source index from dest index using strides,
  or just special-case the shapes you actually use: scalar→shape and
  [k]→[m,k]/[m,1]→[m,n]).
- Sum: loop accumulating into a scalar; SumAxis: nested loops.

### 6.3 matmul via imported kernel (HARD)
```rust
pub extern "C" fn mj_matmul(a: *const f64, b: *const f64, out: *mut f64, m: i64, k: i64, n: i64) { /* triple loop */ }
```
Emit a `call` to the imported `mj_matmul` FuncId with the child buffer pointers
and shape constants. Import transcendental libm funcs the same way
(`mj_exp(x)->f64`, etc.) so elementwise exp/log/tanh/sigmoid also become calls.

### 6.4–6.5 Finalize & call
- `module.define_function(id, &mut ctx)`, `module.finalize_definitions().unwrap()`.
- `let code = module.get_finalized_function(id);`
- `let f = unsafe { std::mem::transmute::<_, extern "C" fn(*const *const f64, *mut *mut f64)>(code) };`
- Host wrapper `pub fn jit_compile(g, out) -> CompiledFn` returns a closure
  capturing the module (keep `JITModule` alive!) + buffer plan; calling it
  allocates buffers, copies inputs, invokes `f`, reads the output buffer into a
  `Tensor`.

### 6.6–6.7 Tests & bench
- `jit_matches_interp` across elementwise/matmul/full-loss graphs (tol 1e-9).
- Benchmark a big matmul-heavy graph JIT vs interp; assert JIT ≥ interp speed
  (or just print the ratio). Real bench goes in Phase 14 criterion harness.

**Gotchas:** keep `JITModule` owned by the returned struct (dropping it frees the
code). f64 in CLIF is `types::F64`. Pointer arithmetic: `iadd` on the base with
`imul_imm(i, 8)`. Verify the exact 0.115 `InstBuilder` method names when you
start (they're stable-ish: `fadd`, `fmul`, `fsub`, `fdiv`, `fmax`, `load`,
`store`, `iadd`, `imul_imm`, `icmp`, `brif`/`brz`, `jump`).

Done-criteria: JIT == interp == VM on all graphs; measured Nx speedup on matmul.

---

## 6. Phase 7 — Widen the op set

Add these ops end-to-end: **forward (interp) + backward (autodiff) + gradient
test**, and (where it makes sense) VM + JIT + optimizer support. `Op::Softmax`,
`CrossEntropy` already have interp forward; the rest are `unimplemented!`.

Order of difficulty (do in this order):

### 7.1 softmax + cross-entropy backward
- Softmax forward exists. Backward: for `y = softmax(x)`, `dx = y * (dy - sum(dy*y))`.
  Implement as graph nodes: needs `Mul`, `Sub`, `Sum`, `Broadcast`. Add an
  `Op::Softmax` arm to `autodiff::grad` building those nodes (the softmax output
  node id `id` gives `y`).
- CrossEntropy(logits, labels) forward exists (returns scalar). Backward wrt
  logits: `softmax(logits) - labels` (times upstream scalar grad). wrt labels:
  usually not needed — return zeros or skip. Gradient-test with the numeric
  harness (labels held fixed as an input; check grad wrt logits).

### 7.2 conv2d + backward (the big one, HARD)
- Represent tensors NCHW-flat in `ArrayD` (shape `[N,C,H,W]`). Add
  `Op::Conv2d{stride,pad}` interp: input `[N,Cin,H,W]`, weight `[Cout,Cin,KH,KW]`
  → `[N,Cout,OH,OW]`. Implement the direct 6-nested-loop convolution first
  (correctness over speed). Shapes: `OH=(H+2pad-KH)/stride+1`.
- Builder `g.conv2d(x, w, stride, pad)` with shape inference; store stride/pad in
  the op. Decide bias handling (fold in as a separate broadcast-add for now).
- Backward: three grads.
  - `dL/dx` = full (transposed) convolution of `dy` with the flipped weight.
  - `dL/dw` = correlation of input patches with `dy`.
  - Implement these as **new interp ops** `Conv2dInputGrad`/`Conv2dWeightGrad`
    (simplest: dedicated backward ops with their own kernels), OR express conv as
    im2col + matmul so autodiff reuses matmul's backward. **Recommendation for a
    weaker model: use im2col.** `im2col(x) -> [N*OH*OW, Cin*KH*KW]`, reshape `w`
    to `[Cin*KH*KW, Cout]`, `matmul`, reshape to `[N,Cout,OH,OW]`. Then autodiff
    is automatic (im2col + matmul + reshape all already differentiable — you only
    need to add `Im2Col` op + its backward `col2im`). This is far less error-prone
    than hand-writing conv backward.
- Gradient-test conv on a tiny case (`N=1,Cin=1,H=W=4,Cout=1,K=3,stride=1,pad=0`).

### 7.3 max-pool + backward
- `Op::MaxPool{kernel,stride}` forward: windowed max. Backward: route the
  upstream grad to the argmax position in each window (store argmax indices during
  forward, or recompute in backward). Add an interp op that carries the mask.
  Gradient-test (note: max is non-differentiable at ties; pick inputs with no
  ties so the numeric check is stable).

### 7.4 batchnorm + backward
- Forward (training mode): `y = (x - mean) / sqrt(var + eps) * gamma + beta`,
  mean/var over the batch axis. Backward is the standard BN gradient — or, again,
  **compose from primitives** (sub, mul, div, sum, broadcast, a new `Rsqrt` or
  `sqrt` op) so autodiff handles it. Add `Op::Sqrt` (grad `0.5/sqrt(x)`) and
  build BN as a subgraph in a builder helper `g.batchnorm(x, gamma, beta, eps)`.
  This avoids a bespoke backward.

### 7.5 tanh, sigmoid — already present. Add any remaining elementwise you want
(`Sqrt` from 7.4; maybe `Abs`).

### 7.6 optimizer + cost model for new ops
- Only add `Mjx` variants for ops you actually want the optimizer to reason about
  (e.g. don't bother fusing conv yet). At minimum add `Softmax`/`Sqrt` if you
  want them to survive round-trip; otherwise the optimizer will `panic!` on graphs
  containing them. **Simplest:** make `opt::to_recexpr` return a `Result` and have
  `optimize` skip optimization (return the graph unchanged) when it hits an
  unsupported op, instead of panicking. That keeps the optimizer safe on the full
  op set.

### 7.7 gradient-check ALL new ops (the gate).

Done-criteria: ~30 ops total, every one gradient-checked, interp supports all,
backends support the core subset.

---

## 7. Phase 8 — NN framework + training (`src/nn.rs`)

**Goal:** train a real net end-to-end on your stack.

### 8.1 Layer abstraction
```rust
pub trait Layer {
    // append this layer's forward subgraph to g; return output node.
    fn forward(&self, g: &mut Graph, x: NodeId) -> NodeId;
    fn params(&self) -> Vec<ParamId>;   // handles into a Params store
}
```
- `struct Params { tensors: Vec<Tensor>, shapes: Vec<Vec<usize>> }` holds the
  live parameter values. A layer references params by index. When building the
  loss graph, each param becomes an `Op::Input` slot whose runtime value comes
  from `Params`.
- Layers: `Linear{in,out}` (weight+bias), `Relu`, `Tanh`, `Sigmoid`,
  `Conv2d{...}`, `MaxPool{...}`, `BatchNorm`, `Softmax`/`CrossEntropy` head.

### 8.2 Parameter init
- Xavier/He init using `rand`: `w ~ U(-lim, lim)`, `lim = sqrt(6/(fan_in+fan_out))`.
  Deterministic seed for reproducible tests.

### 8.3 Optimizers
- `trait Optimizer { fn step(&mut self, params: &mut Params, grads: &[Tensor]); }`
- `Sgd{lr}`: `p -= lr * g`.
- `Adam{lr,b1,b2,eps, m, v, t}`: standard update; store per-param moment tensors.

### 8.4 Data
- `two_moons(n, noise, seed) -> (X[n,2], y[n])` — generate synthetically (two
  interleaving half-circles). Pure Rust + `rand`. This is the primary demo (no
  file IO).
- MNIST loader: optional. Read the IDX format from disk if files present; else
  skip. Don't block the phase on MNIST — two-moons is enough for the gate.

### 8.5 Training loop
- Build the loss graph once (params + batch inputs as `Op::Input` slots). Each
  step: assemble the `inputs` slice from current `Params` + the batch, call
  `grad(g, loss, &param_nodes)`, `eval` to get grad tensors, `optimizer.step`.
- **Perf note:** rebuild-per-step is fine for two-moons. For speed, `eval` once
  computes both loss and grads (grads are just more nodes in the same graph).

### 8.6 Small CNN on MNIST subset (optional if MNIST present).

### 8.7 Log loss/accuracy each epoch (println or return a Vec for the test).

### Tests (`nn::tests`)
- `mlp_learns_two_moons`: train a 2-16-16-2 MLP for K epochs; assert final train
  accuracy > 0.95 (deterministic seed). This is the headline "it actually trains"
  test.
- `adam_decreases_loss`: assert loss strictly decreases over the first N steps.

Done-criteria: an MLP reaches >95% on two-moons on your own autodiff+interp.

---

## 8. Phase 9 — Differential fuzzer (`src/fuzz.rs`)

**Goal:** generate random valid graphs, run them through every backend + the
optimizer, and flag any disagreement; minimize and triage failures. This is what
makes the whole stack trustworthy (and finds real bugs in your VM allocator /
optimizer / JIT).

### 9.1 Random valid-program generator (HARD — shape safety is the crux)
- Maintain a pool of `(NodeId, shape)` "available values", seeded with a few
  random `Input`s and `Const`s of chosen shapes.
- Repeat `k` times: pick an op; pick operand(s) from the pool **whose shapes are
  compatible** (for binary ops, either equal shapes or broadcastable — prefer
  equal to avoid explosion; for matmul, find two 2-D values with matching inner
  dim, or synthesize a compatible new input). Append the node, add to pool.
- Restrict to the "safe" op subset first: Add, Sub, Mul, Relu, Tanh, MatMul,
  Transpose, Sum, Broadcast. Avoid Div (div-by-zero), Log (domain), until the
  oracle is stable. Bound shapes small (dims ≤ 4) and depth small (≤ 20 nodes).
- Final node: wrap in `Sum` to get a scalar (keeps output comparison trivial).
- Return `(Graph, output, input_shapes)`. Generate concrete inputs with values in
  a safe range (e.g. `U(-2,2)`), avoiding exact zeros where an op is sensitive.

### 9.2 Cross-backend oracle
- For a generated program + inputs: compute `interp`, `vm::run`, and (if built)
  `jit`. Assert pairwise `|a-b| < 1e-9` (they should be bit-identical; use a
  small tol for JIT which may use FMA/libm differences). Any mismatch → failure.

### 9.3 Metamorphic oracle (opt vs unopt)
- Compute `eval(orig)` vs `eval(optimize_sound(orig))`. With **sound-only** rules
  the results must match to ~1e-12 (no reassociation). With fast-math `optimize`,
  allow a looser tol (reassociation changes rounding) — or better, assert they
  match over *reals* by comparing at low precision. Use `optimize_sound` for the
  strict oracle; this is exactly where the Phase-4 gating pays off.

### 9.4 Failure minimizer (delta-debug)
- Given a failing `(Graph, inputs)`, repeatedly try to shrink: remove a node and
  re-root at an earlier node; shrink dims; replace subtrees with inputs. Keep any
  reduction that still fails. Standard ddmin-style loop until no single reduction
  still reproduces.

### 9.5 Dedup + triage
- Hash failures by (which oracle, op set involved, error signature). Keep one
  minimal example per bucket.

### 9.6 Campaign
- `pub fn campaign(iters, seed) -> Vec<Failure>`. Run in a test with a fixed seed
  and a modest iter count (e.g. 2000) and **assert zero failures** once the
  backends are correct. During development, expect it to surface real bugs (esp.
  VM buffer-reuse aliasing) — fix them, then lock the test at zero.

Done-criteria: fuzzer runs unattended, minimizes, and the seeded campaign is
green (all backends + sound-optimizer agree).

---

## 9. Phase 10 — Text frontend (`src/frontend/` — new module dir)

**Goal:** run programs from `.mjx` source, not just the Rust builder API.

Suggested files: `frontend/lexer.rs`, `frontend/parser.rs`, `frontend/ast.rs`,
`frontend/lower.rs`, and `frontend/mod.rs`.

### 10.1 Lexer (`logos` — already a dep)
- Tokens: identifiers, number literals, `let = ( ) , [ ] : ;`, keywords
  `input const relu matmul sum ...`, operators `+ - * /`, arrow/colon for shapes.
- `#[derive(Logos)]` enum; `Token::lexer(src)` yields tokens with spans.

### 10.2 Parser → AST
- Recursive descent (don't pull in `chumsky` unless you want to; hand-rolled is
  fine and fewer deps). Grammar sketch:
  ```
  program := stmt* expr
  stmt    := 'let' ident '=' expr ';'
  expr    := term (('+'|'-') term)*
  term    := factor (('*'|'/') factor)*
  factor  := ident | number | call | '(' expr ')'
  call    := ident '(' args ')'          // relu(x), matmul(a,b), input([2,3]), sum(x)
  ```
- AST nodes carry spans for error messages.

### 10.3 Lower AST → IR
- Walk the AST with an environment `HashMap<String, NodeId>`. `input([...])`
  declares a graph input; `const(v, [...])` a constant; calls map to builder
  methods; `+ - * /` to `g.add/sub/mul/div`. Return the final expression's node.

### 10.4 Type/shape checker
- Shape inference already lives in the builder (it panics on mismatch). Upgrade to
  return `Result<NodeId, TypeError>` with a span + message ("matmul inner dim
  mismatch 3 vs 4 at line L"). Either pre-check in the lowerer or convert the
  builder asserts to errors for the frontend path.

### 10.5 `.mjx` example files
- Put 2–3 in `examples/*.mjx` (the loss function, an MLP forward, a matmul chain).

### Tests
- `parses_and_runs_loss_mjx`: load source, lower, eval, compare to the Rust-built
  equivalent. `reports_shape_error`: a bad program yields a nice error.

Done-criteria: programs run from source text; shape errors are readable.

---

## 10. Phase 11 — Second codegen backend: C-emit (`src/cbackend.rs`)

**Goal:** a third backend for cross-validation + a portability story. C-emit is
the least-dep option (no LLVM/wgpu toolchain needed at build time; needs `cc`/
`gcc` at runtime for the test).

### 11.1–11.3
- `pub fn emit_c(g, out) -> String`: generate a C translation unit with one
  `double* mj_run(double** inputs)` function. Per node emit a stack/heap buffer
  and a loop (mirror the JIT lowering but as C source). matmul as an emitted C
  triple loop. Elementwise via `<math.h>` (`expf`? use `exp`, `tanh`).
- Build/link/run pipeline (test only): write the `.c` to the scratchpad, invoke
  `cc -O2 -shared -fPIC -o libmj.so mj.c` via `std::process::Command`, `dlopen`
  it (add `libloading`? or just compile to an executable that reads inputs from
  stdin/argv and prints the result — **simpler: emit a `main` that hardcodes the
  test inputs and prints the output**, run it, parse stdout). Avoid new deps by
  using the executable-prints-result approach.

### 11.4 Differential test + bench
- `c_matches_interp`: guarded by `#[cfg(...)]` / a runtime check that `cc` exists;
  skip gracefully if not. Compare printed output to interp within 1e-9.

Done-criteria: C backend validated against interp; note it in the benches.

---

## 11. Phase 12 — Advanced optimizer passes (`src/memplan.rs`)

**Goal:** measurable memory-traffic reduction via liveness + buffer reuse and a
fusion-scheduling pass. Much of this overlaps Phase 5.3 — factor the analysis so
both VM and this pass use it.

### 12.1 Liveness
- Over the linearized graph (topo order), compute per-node `[def, last_use]`
  intervals (already needed in 5.3). Expose `fn liveness(g, out) -> Vec<(usize,usize)>`.

### 12.2 Memory planner (HARD)
- Linear-scan register allocation over the intervals: sweep instruction index,
  maintain a set of free buffers keyed by byte-size (bucket by shape), assign a
  buffer to each node reusing a freed one of matching size when available.
- Report **peak live bytes** before vs after planning (sum of live buffer sizes at
  the worst instruction). This is the headline metric.
- In-place ops: when a unary/elementwise op's sole input dies at this instruction
  and has the same shape, write in place (dst buffer = src buffer). Detect and
  emit an `in_place` flag.

### 12.3 Operator-fusion scheduling
- Identify chains of elementwise ops with matching shape and single consumers;
  mark them as a fused group (one loop, no intermediate buffers). Even if you
  don't codegen the fusion, report how many intermediate buffers it eliminates.

### 12.4 Layout selection — optional; a cost-model toggle row/col-major for matmul
operands. Can be a stub that reports the chosen layout.

### 12.5 Measure
- A test/bench printing peak-bytes before/after and buffer count before/after on
  a deep MLP graph; assert after < before.

Done-criteria: measurable peak-memory reduction reported.

---

## 12. Phase 13 — Tooling (`src/cli.rs`, `src/viz.rs`)

### 13.1 Graphviz/DOT export (`viz.rs`)
- `pub fn to_dot(g, out) -> String`: nodes labeled by op + shape; edges from
  inputs. A second entry that also renders the gradient graph (build grads, then
  DOT the whole thing, coloring gradient nodes). Write `.dot` to disk; user runs
  `dot -Tpng`.

### 13.2 CLI (`cli.rs`, `clap` derive — already a dep)
Subcommands, each wiring an existing module:
- `run <file.mjx>` — parse, eval, print result.
- `grad <file.mjx> --wrt <inputs>` — print gradients.
- `opt <file.mjx> [--fast-math]` — print op-count + FLOP before/after
  (`opt::optimize` / `optimize_sound`).
- `verify` — print `verify::soundness_report()`.
- `fuzz --iters N --seed S` — run a campaign, print failures.
- `train --dataset two-moons` — run Phase 8 training, print curve.
- `viz <file.mjx> [--grad] -o out.dot` — write DOT.
Replace `main.rs`'s placeholder with `cli::main()` dispatch.

### 13.3 REPL — read a line, lower, eval, print; keep a persistent env.

### 13.4 Profiler — wrap `eval` with per-op timing (accumulate `Duration` per
`Op` discriminant); print a table.

Done-criteria: usable CLI covering every subsystem; DOT output renders.

---

## 13. Phase 14 — Polish, benches, write-up

### 14.1 proptest invariants (dev-dep already present)
- Property: for random valid graphs (reuse the fuzzer generator), `optimize_sound`
  preserves eval within 1e-10. Property: reverse-mode grad == forward-mode
  directional derivative (generalize the existing cross-check).

### 14.2 criterion benches (`benches/bench.rs`)
- interp vs VM vs JIT vs C on a matmul-chain and an MLP forward+backward.
- optimizer: op-count and FLOP reduction on representative graphs.
- Fill in the real benches replacing the placeholder.

### 14.3 README
- Architecture diagram (ASCII or the DOT output), the headline metrics (JIT
  speedup, optimizer FLOP reduction, memory-planner peak-byte reduction, fuzzer
  iterations green), and **the float-soundness finding** with the printed
  counterexample table from `verify::soundness_report()`.

### 14.4 Design doc
- Short: the AD design (arena + reverse topo), egg round-trip (why shapes are
  encoded as nodes + re-inferred), the Z3 reals-vs-IEEE encoding and the finite
  trick, the fuzzer oracles. Capture the reasoning in §2 gotchas so it isn't lost.

Done-criteria: clean repo, `just check` green (fmt+clippy+test), headline numbers,
write-up done.

---

## 14. Suggested build order & checkpoints

1. **Finish Phase 4** (confirm tests + add `optimize_sound` gating). ← smallest
   remaining unit; unblocks the fuzzer's strict oracle.
2. **Phase 5 VM** — pure Rust, no new toolchain; gives the fuzzer a second oracle.
3. **Phase 9 fuzzer (interp vs VM + metamorphic)** early — even before JIT. It
   will immediately harden the VM allocator and the optimizer. Re-run after every
   later backend.
4. **Phase 7 op widening** (via im2col/primitive-composition to keep autodiff
   automatic) — needed before real training.
5. **Phase 8 NN training** — the "wow" demo; depends on 7.
6. **Phase 6 JIT** — highest-risk, highest-reward; add it as a third fuzzer oracle.
7. **Phases 10–13** (frontend, C-emit, memory planner, CLI/viz) — independent,
   any order.
8. **Phase 14** — polish + write-up last.

Stop-anywhere: the project is demoable after Phase 8 (train a net on your own
autodiff + verified optimizer). Everything after is depth.

### Per-phase definition of done
A phase is done only when: its module compiles clean under
`cargo clippy -- -D warnings`, its tests pass, AND (for backends/ops) the
differential test against `interp` is green. Update the status table in §0 and
tick the boxes in `PLAN.md` as you go.
