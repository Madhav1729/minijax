# C++ port roadmap for minijax

This document tracks the C++ version of the Rust project. The target is a full end-to-end differentiable tensor compiler in roughly the same conceptual shape, with an implementation budget around 12k-13k lines of code when all phases are complete.

## Core phases

### 1. Core tensor + graph IR
- tensor shape utilities and broadcasting
- node arena and operation set
- graph builder for binary ops, matmul, reductions

### 2. Interpreter and correctness checks
- evaluate graph in topo order
- compare against hand-computed tests
- keep interpreter as the reference implementation

### 3. Reverse-mode autodiff
- gradient accumulation
- backward rules for add/mul/matmul/relu/sum/broadcast
- numerical grad-check harness

### 4. Optimizer layer
- rewrite rules
- cost model
- equality-saturation / rewrite set framework
- correctness validation against interpreter

### 5. SMT verification layer
- rule proof generation
- real-vs-float classification
- counterexample handling

### 6. Bytecode VM backend
- compile graph to instruction stream
- register allocation
- executor matching interpreter output

### 7. JIT backend
- Cranelift-like lowering strategy or optimized CPU backend
- fused vector kernels
- benchmark + differential tests

### 8. NN framework
- Linear, activation, loss layers
- parameter storage and optimizer steps
- minimal training loop

### 9. Fuzzer and differential testing
- random valid programs
- compare interpreter vs VM vs JIT
- crash minimization and regression tracking

### 10. Frontend and CLI
- basic text parser
- REPL + subcommands
- graph visualization

## Deliverables

- stable C++ build with test suite
- reproducible graph evaluation
- AD gradient flow
- backend differential validation
- CLI entrypoint for core commands

This is the roadmap to get from the current working prototype to the full project equivalent of the Rust implementation.
