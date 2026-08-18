#include <chrono>
#include <iostream>
#include <vector>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

using namespace minijax;

static double time_eval(const Graph& g, const std::vector<Tensor>& inputs, size_t iters = 10) {
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        auto out = eval(g, inputs);
        (void)out;
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    const int N = 96;
    Shape s = {N, N};

    Graph g;
    auto a = g.input(s);
    auto b = g.input(s);
    auto c = g.input(s);
    auto m1 = g.matmul(a, b);
    auto m2 = g.matmul(m1, c);
    for (int i = 0; i < 2; ++i) {
        m2 = g.matmul(m2, a);
    }
    auto out = g.sum(m2);

    Tensor A(s, std::vector<double>(N * N, 1.0));
    Tensor B(s, std::vector<double>(N * N, 2.0));
    Tensor C(s, std::vector<double>(N * N, 0.5));
    std::vector<Tensor> inputs = {A, B, C};

    Graph base = g;
    Graph opt_g = g;
    rebuild_with_assoc(opt_g);
    optimize_until_fixedpoint(opt_g);
    extract_by_cost(opt_g);

    double t_base = time_eval(base, inputs, 3);
    double t_opt = time_eval(opt_g, inputs, 3);

    std::cout << "microbench_large: baseline=" << t_base << "s optimized=" << t_opt << "s\n";
    return 0;
}
