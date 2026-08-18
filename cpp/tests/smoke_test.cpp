#include <cmath>
#include <iostream>

#include "autodiff.hpp"
#include "interp.hpp"

int main() {
    minijax::Graph g;
    size_t w = g.input({2, 2});
    size_t x = g.input({2, 1});
    size_t y = g.input({2, 1});

    size_t wx = g.matmul(w, x);
    size_t act = g.relu(wx);
    size_t diff = g.sub(act, y);
    size_t sq = g.mul(diff, diff);
    size_t loss = g.sum(sq);

    minijax::Tensor W = minijax::Tensor::from_matrix({{1.0, 0.0}, {0.0, 1.0}});
    minijax::Tensor X = minijax::Tensor::from_matrix({{2.0}, {3.0}});
    minijax::Tensor Y = minijax::Tensor::from_matrix({{1.0}, {1.0}});

    auto values = minijax::eval(g, {W, X, Y});
    double loss_value = values[loss].scalar();
    if (std::abs(loss_value - 5.0) > 1e-9) {
        std::cerr << "expected loss 5, got " << loss_value << std::endl;
        return 1;
    }

    auto grads = minijax::grad(g, loss, {w, x, y});
    auto grad_values = minijax::eval(g, {W, X, Y});
    std::cout << "loss=" << loss_value << " grad_count=" << grads.size() << std::endl;
    (void)grad_values;
    return 0;
}
