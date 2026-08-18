#include <chrono>
#include <iostream>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

using namespace minijax;

double time_eval(const Graph& g, const std::vector<Tensor>& inputs, size_t iters=50) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        auto out = eval(g, inputs);
        (void)out;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> d = end - start;
    return d.count();
}

int main() {
    // run two sizes to compare behavior: N=64 and N=256
    for (int N : {64, 256}) {
        Graph g;
        Shape s = {N, N};
        size_t a = g.input(s);
        size_t b = g.input(s);
        size_t t = g.matmul(a, b);
        for (int i = 0; i < 3; ++i) t = g.matmul(t, a);
        size_t sumn = g.sum(t);

        Tensor A(s, std::vector<double>(N*N, 1.0));
        Tensor B(s, std::vector<double>(N*N, 2.0));
        std::vector<Tensor> inputs = {A, B};

        Graph g_base = g;
        double t_base = time_eval(g_base, inputs, (N<=64) ? 5 : 2);

        Graph g_flop = g;
        rebuild_with_assoc(g_flop);
        optimize_until_fixedpoint(g_flop);
        extract_by_flops(g_flop);
        double t_flop = time_eval(g_flop, inputs, (N<=64) ? 5 : 2);

        Graph g_cost = g;
        rebuild_with_assoc(g_cost);
        optimize_until_fixedpoint(g_cost);
        extract_by_cost(g_cost);
        double t_cost = time_eval(g_cost, inputs, (N<=64) ? 5 : 2);

        std::cout << "microbench N=" << N << ": baseline=" << t_base << "s flop_extract=" << t_flop << "s cost_extract=" << t_cost << "s\n";
    }
    return 0;
}
