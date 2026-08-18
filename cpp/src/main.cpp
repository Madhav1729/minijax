#include <iostream>
#include <vector>

#include "autodiff.hpp"
#include "interp.hpp"

namespace minijax {

void print_loss_example() {
    Graph g;
    size_t w = g.input({2, 2});
    size_t x = g.input({2, 1});
    size_t y = g.input({2, 1});

    size_t wx = g.matmul(w, x);
    size_t act = g.relu(wx);
    size_t diff = g.sub(act, y);
    size_t sq = g.mul(diff, diff);
    size_t loss = g.sum(sq);

    Tensor W = Tensor::from_matrix({{1.0, 0.0}, {0.0, 1.0}});
    Tensor X = Tensor::from_matrix({{2.0}, {3.0}});
    Tensor Y = Tensor::from_matrix({{1.0}, {1.0}});

    std::vector<Tensor> inputs = {W, X, Y};
    auto values = eval(g, inputs);
    std::cout << "loss = " << values[loss].scalar() << std::endl;

    auto grads = grad(g, loss, {w, x, y});
    auto grad_values = eval(g, inputs);
    std::cout << "grad_w = " << grad_values[grads[0]].to_string() << std::endl;
    std::cout << "grad_x = " << grad_values[grads[1]].to_string() << std::endl;
    std::cout << "grad_y = " << grad_values[grads[2]].to_string() << std::endl;
}

}  // namespace minijax

int main() {
    minijax::print_loss_example();
    return 0;
}
