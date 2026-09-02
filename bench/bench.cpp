


#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "minijax/interp.hpp"
#include "minijax/vm.hpp"
#include "minijax/opt.hpp"
#include "minijax/autodiff.hpp"
#include "minijax/fuzz.hpp"
#include "minijax/nn.hpp"
#if defined(MINIJAX_WITH_JIT)
#include "minijax/jit.hpp"
#endif

using namespace minijax;

namespace {

using clock_ = std::chrono::steady_clock;

template <class F>
double time_ms(F&& f, int iters) {
    f();
    auto t0 = clock_::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = clock_::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

size_t numel_of(const std::vector<size_t>& shape) {
    return shape.empty() ? 1
                         : std::accumulate(shape.begin(), shape.end(), size_t{1},
                                           [](size_t a, size_t b) { return a * b; });
}


double flop_estimate(const Graph& g, NodeId out) {
    std::vector<bool> seen(g.size(), false);
    std::vector<NodeId> stack = {out};
    double total = 0;
    while (!stack.empty()) {
        NodeId id = stack.back();
        stack.pop_back();
        if (seen[id]) continue;
        seen[id] = true;
        const Node& n = g.node(id);
        size_t n_out = numel_of(n.shape);
        switch (n.op) {
            case OpKind::MatMul: {
                const auto& a = g.shape_of(n.inputs[0]);
                const auto& b = g.shape_of(n.inputs[1]);
                total += 2.0 * static_cast<double>(a[0] * a[1] * b[1]);
                break;
            }
            case OpKind::Input:
            case OpKind::Const:
                break;
            case OpKind::Reshape:
                total += 0;
                break;
            default:
                total += static_cast<double>(n_out);
        }
        for (NodeId in : n.inputs) stack.push_back(in);
    }
    return total;
}

Graph make_matmul_chain(size_t dim, int depth, NodeId& out) {
    Graph g;
    NodeId cur = g.input({dim, dim});
    NodeId w = g.constant(0.999, {dim, dim});
    for (int i = 0; i < depth; ++i) cur = g.matmul(cur, w);
    out = g.sum(cur);
    return g;
}

Graph make_elementwise_chain(size_t n, int depth, NodeId& out) {
    Graph g;
    NodeId cur = g.input({n});
    for (int i = 0; i < depth; ++i) {
        switch (i % 4) {
            case 0: cur = g.tanh(cur); break;
            case 1: cur = g.exp(cur); break;
            case 2: cur = g.add(cur, cur); break;
            default: cur = g.mul(cur, g.constant(0.5)); break;
        }
    }
    out = g.sum(cur);
    return g;
}

}

int main(int argc, char** argv) {
    int iters = 50;
    if (argc > 1) iters = std::atoi(argv[1]);

    std::cout << "minijax-cpp bench (" << iters << " iterations per measurement)\n";


    {
        constexpr size_t D = 32;
        constexpr int DEPTH = 30;
        NodeId out;
        Graph g = make_matmul_chain(D, DEPTH, out);
        std::vector<Tensor> inputs = {Tensor::from_vec({D, D},
            std::vector<double>(D * D, 0.001))};

        Program prog = compact(compile(g, out));
        double t_i = time_ms([&] { (void)eval(g, inputs, out); }, iters);
        double t_v = time_ms([&] { (void)run_vm(prog, inputs); }, iters);
        double t_j = std::nan("");
#if defined(MINIJAX_WITH_JIT)
        {
            jit::JitProgram jp(g);
            t_j = time_ms([&] { jp.execute(inputs); }, iters);
        }
#endif
        std::cout << "\n[1] matmul chain  dim=" << D << " depth=" << DEPTH << "\n";
        std::cout << "  backend                     ms/call\n";
        std::printf("  %-22s %10.3f\n", "interpreter", t_i);
        std::printf("  %-22s %10.3f\n", "vm (compacted)", t_v);
        if (!std::isnan(t_j)) std::printf("  %-22s %10.3f\n", "jit", t_j);
        if (!std::isnan(t_j) && t_j > 0)
            std::printf("  jit speedup vs interp: %.2fx\n", t_i / t_j);
    }


    {
        constexpr size_t N = 4096;
        constexpr int DEPTH = 64;
        NodeId out;
        Graph g = make_elementwise_chain(N, DEPTH, out);
        std::vector<Tensor> inputs = {Tensor::from_vec({N},
            std::vector<double>(N, 0.5))};
        Program prog = compact(compile(g, out));
        double t_i = time_ms([&] { (void)eval(g, inputs, out); }, iters);
        double t_v = time_ms([&] { (void)run_vm(prog, inputs); }, iters);
        double t_j = std::nan("");
#if defined(MINIJAX_WITH_JIT)
        {
            jit::JitProgram jp(g);
            t_j = time_ms([&] { jp.execute(inputs); }, iters);
        }
#endif
        std::cout << "\n[2] elementwise chain  n=" << N << " depth=" << DEPTH << "\n";
        std::printf("  %-22s %10.3f\n", "interpreter", t_i);
        std::printf("  %-22s %10.3f\n", "vm (compacted)", t_v);
        if (!std::isnan(t_j)) {
            std::printf("  %-22s %10.3f\n", "jit", t_j);
            if (t_i > 0 && t_j > 0)
                std::printf("  jit speedup vs interp: %.2fx\n", t_i / t_j);
        }
    }


    {
        Dataset data = generate_two_moons(120, 0.08, 7);
        Graph g;
        NodeId x_node = g.input({2, 1});
        NodeId y_node = g.input({2, 1});
        MLP mlp(g, {2, 16, 16, 2}, 42);
        NodeId logits = mlp.forward(g, x_node);
        NodeId probs = g.softmax(logits);
        NodeId y_flat = g.reshape(y_node, {2});
        NodeId loss = g.cross_entropy(probs, y_flat);
        std::vector<NodeId> param_nodes;
        for (const auto& p : mlp.params) param_nodes.push_back(p.node);
        auto grads = grad(g, loss, param_nodes);
        Adam opt;
        opt.lr = 0.005;

        double ms_epoch = time_ms([&] {
            (void)train_epoch(g, mlp.params, x_node, y_node, loss, grads, data, opt);
        }, 5);
        std::printf("\n[3] MLP{2,16,16,2} forward+backward+step\n");
        std::printf("  %.3f ms/epoch (%zu samples) => %.0f samples/s\n",
                    ms_epoch, data.xs.size(),
                    ms_epoch > 0 ? data.xs.size() / (ms_epoch / 1000.0) : 0.0);
    }


    {
        struct Case {
            const char* name;
            Graph g;
            NodeId out;
        };
        std::vector<Case> cases;

        {

            Graph g;
            NodeId x = g.input({8, 8});
            NodeId w = g.constant(0.3, {8, 8});
            NodeId h = g.relu(g.matmul(x, w));
            NodeId h2 = g.mul(h, g.constant(1.0, {8, 8}));
            NodeId h3 = g.add(h2, g.constant(0.0, {8, 8}));
            NodeId o = g.sum(h3);
            cases.push_back({"mul-one/add-zero", std::move(g), o});
        }
        {

            Graph g;
            NodeId x = g.input({64});
            NodeId s1 = g.add(x, x);
            NodeId s2 = g.add(s1, x);
            NodeId s3 = g.add(s2, x);
            NodeId o = g.sum(s3);
            cases.push_back({"add assoc chain", std::move(g), o});
        }
        {

            auto prog = generate_random_program(11, 12, 6);
            cases.push_back({"fuzz seed 11 d12", std::move(prog.g), prog.output});
        }

        std::printf("\n[4] optimizer FLOP-reduction table (heuristic cost model)\n");
        std::printf("  %-20s %8s %8s %12s %12s %8s\n", "graph", "nodes", "->",
                    "MFLOP", "->MFLOP", "saved");
        for (auto& c : cases) {
            size_t nb = c.g.size();
            double fb = flop_estimate(c.g, c.out);
            auto [og, oo] = optimize(c.g, c.out);
            size_t na = og.size();
            double fa = flop_estimate(og, oo);
            double saved = fb > 0 ? 100.0 * (1.0 - fa / fb) : 0.0;
            std::printf("  %-20s %8zu %7zu %12.3f %12.3f %7.1f%%\n",
                        c.name, nb, na, fb / 1e6, fa / 1e6, saved);

            std::vector<Tensor> dummy_inputs;
            for (NodeId id : c.g.inputs()) dummy_inputs.push_back(Tensor::zeros(c.g.shape_of(id)));
            (void)eval(c.g, dummy_inputs, c.out);
            (void)eval(og, dummy_inputs, oo);
        }
        std::cout << "\nall optimized outputs evaluated without error\n";
    }

    return 0;
}
