#include "minijax/fuzz.hpp"
#include "minijax/interp.hpp"
#include "minijax/vm.hpp"
#include "minijax/opt.hpp"
#if defined(MINIJAX_FUZZ_HAS_JIT_ORACLE)
#include "minijax/jit.hpp"
#endif
#include <random>
#include <algorithm>
#include <set>
#include <sstream>
#include <functional>

namespace minijax {

namespace {

enum class GenOp { Add, Sub, Mul, Neg, Relu, Tanh, Sigmoid, Abs, MatMul, Transpose, Sum, SumAxis };

double rand_val(std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    return dist(rng);
}

std::vector<size_t> rand_shape(std::mt19937& rng, int max_dim) {
    std::uniform_int_distribution<int> rank_dist(1, 2);
    std::uniform_int_distribution<int> dim_dist(1, max_dim);
    int rank = rank_dist(rng);
    std::vector<size_t> shape;
    for (int i = 0; i < rank; ++i) shape.push_back(static_cast<size_t>(dim_dist(rng)));
    return shape;
}

}

GeneratedProgram generate_random_program(unsigned seed, int max_depth, int max_dim, int num_inputs) {
    std::mt19937 rng(seed);
    GeneratedProgram result;
    Graph& g = result.g;

    std::vector<NodeId> pool;
    for (int i = 0; i < num_inputs; ++i) {
        std::vector<size_t> shape = rand_shape(rng, max_dim);
        NodeId id = g.input(shape);
        pool.push_back(id);
        result.inputs.push_back(Tensor::zeros(shape).mapv([&](double) { return rand_val(rng); }));
    }

    std::uniform_int_distribution<int> op_dist(0, 11);
    std::uniform_int_distribution<size_t> pick_dist(0, pool.size() - 1);

    for (int step = 0; step < max_depth; ++step) {
        GenOp op = static_cast<GenOp>(op_dist(rng));
        std::uniform_int_distribution<size_t> pick(0, pool.size() - 1);
        NodeId a = pool[pick(rng)];
        NodeId b = pool[pick(rng)];

        try {
            NodeId out = kInvalidNode;
            switch (op) {
                case GenOp::Add: out = g.add(a, b); break;
                case GenOp::Sub: out = g.sub(a, b); break;
                case GenOp::Mul: out = g.mul(a, b); break;
                case GenOp::Neg: out = g.neg(a); break;
                case GenOp::Relu: out = g.relu(a); break;
                case GenOp::Tanh: out = g.tanh(a); break;
                case GenOp::Sigmoid: out = g.sigmoid(a); break;
                case GenOp::Abs: out = g.abs(a); break;
                case GenOp::MatMul: out = g.matmul(a, b); break;
                case GenOp::Transpose: out = g.transpose(a); break;
                case GenOp::Sum: out = g.sum(a); break;
                case GenOp::SumAxis: {
                    const auto& shape = g.shape_of(a);
                    if (shape.empty()) throw std::invalid_argument("skip: rank-0 has no axis");
                    std::uniform_int_distribution<size_t> axis_dist(0, shape.size() - 1);
                    out = g.sum_axis(a, axis_dist(rng));
                    break;
                }
            }
            pool.push_back(out);
        } catch (const std::invalid_argument&) {


        }
    }

    result.output = pool.back();
    return result;
}

OracleResult check_interp_vs_vm(const Graph& g, NodeId output, const std::vector<Tensor>& inputs, double tol) {
    Tensor expected = eval(g, inputs, output);
    Program prog = compact(compile(g, output));
    Tensor got = run_vm(prog, inputs);
    if (!Tensor::allclose(expected, got, tol, tol)) {
        std::ostringstream os;
        os << "interp/vm mismatch at output shape rank " << expected.rank();
        return {false, os.str()};
    }
    return {true, ""};
}

OracleResult check_metamorphic_optimize_sound(const Graph& g, NodeId output, const std::vector<Tensor>& inputs,
                                               double tol) {
    Tensor before = eval(g, inputs, output);
    auto [optimized, new_output] = optimize_sound(g, output);
    Tensor after = eval(optimized, inputs, new_output);
    if (!Tensor::allclose(before, after, tol, tol)) {
        std::ostringstream os;
        os << "optimize_sound() changed the result: before/after mismatch at rank " << before.rank();
        return {false, os.str()};
    }
    return {true, ""};
}

#if defined(MINIJAX_FUZZ_HAS_JIT_ORACLE)
OracleResult check_interp_vs_jit(const Graph& g, NodeId output, const std::vector<Tensor>& inputs, double tol) {
    Tensor expected = eval(g, inputs, output);


    Tensor got = jit::run_jit(g, inputs, output);
    if (!Tensor::allclose(expected, got, tol, tol)) {
        std::ostringstream os;
        os << "interp/jit mismatch at output shape rank " << expected.rank();
        return {false, os.str()};
    }
    return {true, ""};
}
#endif

namespace {


std::string graph_signature(const Graph& g, NodeId output) {
    std::set<std::string> ops;

    std::vector<bool> visited(g.size(), false);
    std::vector<NodeId> stack = {output};
    while (!stack.empty()) {
        NodeId id = stack.back();
        stack.pop_back();
        if (visited[id]) continue;
        visited[id] = true;
        ops.insert(op_kind_name(g.node(id).op));
        for (NodeId in : g.node(id).inputs) stack.push_back(in);
    }
    std::ostringstream os;
    for (const auto& o : ops) os << o << ",";
    return os.str();
}


int minimize_depth(unsigned seed, int original_depth, int max_dim,
                    const std::function<bool(const Graph&, NodeId, const std::vector<Tensor>&)>& still_fails) {
    for (int d = 0; d <= original_depth; ++d) {
        auto prog = generate_random_program(seed, d, max_dim);
        if (still_fails(prog.g, prog.output, prog.inputs)) return d;
    }
    return original_depth;
}

}

CampaignResult run_fuzz_campaign(unsigned base_seed, int iterations, int max_depth, int max_dim,
                                  bool with_jit_oracle) {
    CampaignResult result;
    result.iterations_run = iterations;
    std::set<std::string> seen_signatures;

#if !defined(MINIJAX_FUZZ_HAS_JIT_ORACLE)
    (void)with_jit_oracle;
#endif

    for (int i = 0; i < iterations; ++i) {
        unsigned seed = base_seed + static_cast<unsigned>(i);
        GeneratedProgram prog = generate_random_program(seed, max_depth, max_dim);

        OracleResult r1 = check_interp_vs_vm(prog.g, prog.output, prog.inputs);
        if (!r1.passed) {
            int min_depth = minimize_depth(seed, max_depth, max_dim,
                [&](const Graph& g, NodeId out, const std::vector<Tensor>& in) {
                    return !check_interp_vs_vm(g, out, in).passed;
                });
            auto minimized = generate_random_program(seed, min_depth, max_dim);
            std::string sig = "interp_vs_vm:" + graph_signature(minimized.g, minimized.output);
            if (seen_signatures.insert(sig).second) {
                result.failures.push_back({seed, min_depth, "interp_vs_vm", r1.detail, sig});
            }
        }

        OracleResult r2 = check_metamorphic_optimize_sound(prog.g, prog.output, prog.inputs);
        if (!r2.passed) {
            int min_depth = minimize_depth(seed, max_depth, max_dim,
                [&](const Graph& g, NodeId out, const std::vector<Tensor>& in) {
                    return !check_metamorphic_optimize_sound(g, out, in).passed;
                });
            auto minimized = generate_random_program(seed, min_depth, max_dim);
            std::string sig = "metamorphic:" + graph_signature(minimized.g, minimized.output);
            if (seen_signatures.insert(sig).second) {
                result.failures.push_back({seed, min_depth, "metamorphic_optimize_sound", r2.detail, sig});
            }
        }

#if defined(MINIJAX_FUZZ_HAS_JIT_ORACLE)
        if (with_jit_oracle) {
            OracleResult r3 = check_interp_vs_jit(prog.g, prog.output, prog.inputs);
            if (!r3.passed) {
                int min_depth = minimize_depth(seed, max_depth, max_dim,
                    [&](const Graph& g, NodeId out, const std::vector<Tensor>& in) {
                        return !check_interp_vs_jit(g, out, in).passed;
                    });
                auto minimized = generate_random_program(seed, min_depth, max_dim);
                std::string sig = "interp_vs_jit:" + graph_signature(minimized.g, minimized.output);
                if (seen_signatures.insert(sig).second) {
                    result.failures.push_back({seed, min_depth, "interp_vs_jit", r3.detail, sig});
                }
            }
        }
#endif
    }

    return result;
}

}
