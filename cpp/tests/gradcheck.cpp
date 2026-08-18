#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "autodiff.hpp"
#include "interp.hpp"
#include "ir.hpp"
#include "tensor.hpp"

namespace {

void expect_close(double actual, double expected, double tol, const std::string& msg) {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(msg + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

// Reuse the finite-difference helper
double finite_diff_scalar(const minijax::Graph& g,
                         const std::vector<minijax::Tensor>& inputs,
                         size_t input_index,
                         size_t input_elem,
                         size_t scalar_output,
                         double eps = 1e-6) {
    minijax::Tensor x = inputs[input_index];
    minijax::Tensor xp = x;
    xp.data[input_elem] += eps;

    minijax::Tensor xm = x;
    xm.data[input_elem] -= eps;

    auto fp = minijax::eval(g, [&]() {
        std::vector<minijax::Tensor> arr = inputs;
        arr[input_index] = xp;
        return arr;
    }());
    auto fm = minijax::eval(g, [&]() {
        std::vector<minijax::Tensor> arr = inputs;
        arr[input_index] = xm;
        return arr;
    }());

    return (fp[scalar_output].scalar() - fm[scalar_output].scalar()) / (2.0 * eps);
}

void check_matmul_grad() {
    minijax::Graph g;
    auto W = g.input({2, 3});
    auto X = g.input({3, 2});
    auto Y = g.matmul(W, X);
    auto s = g.sum(Y);

    minijax::Tensor Tw = minijax::Tensor::from_matrix({{1.0, 2.0, -1.0}, {0.5, -0.5, 2.0}});
    minijax::Tensor Tx = minijax::Tensor::from_matrix({{1.0, 0.0}, {0.0, 1.0}, {2.0, -1.0}});

    auto grads = minijax::grad(g, s, {W, X});
    auto vals = minijax::eval(g, {Tw, Tx});

    // Numerical checks
    std::vector<minijax::Tensor> inputs = {Tw, Tx};
    for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
        const auto& t = inputs[in_idx];
        for (size_t e = 0; e < t.numel(); ++e) {
            double fd = finite_diff_scalar(g, inputs, in_idx, e, s);
            auto grad_tensor = vals[grads[in_idx]];
            double gval = grad_tensor.at_1d(e);
            expect_close(gval, fd, 1e-6, "matmul grad numeric mismatch");
        }
    }
}

void check_broadcast_add_grad() {
    minijax::Graph g;
    auto a = g.input({2, 1});
    auto b = g.input({2, 2});
    auto aa = g.broadcast(a, {2, 2});
    auto c = g.add(aa, b);
    auto s = g.sum(c);

    minijax::Tensor Ta({2, 1}, {1.5, -2.0});
    minijax::Tensor Tb({2, 2}, {0.5, 1.0, -1.0, 2.0});

    auto grads = minijax::grad(g, s, {a, b});
    auto vals = minijax::eval(g, {Ta, Tb});

    std::vector<minijax::Tensor> inputs = {Ta, Tb};
    for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
        const auto& t = inputs[in_idx];
        for (size_t e = 0; e < t.numel(); ++e) {
            double fd = finite_diff_scalar(g, inputs, in_idx, e, s);
            auto grad_tensor = vals[grads[in_idx]];
            double gval = grad_tensor.at_1d(e);
            expect_close(gval, fd, 1e-6, "broadcast add grad numeric mismatch");
        }
    }
}

} // namespace

int main() {
    try {
        check_matmul_grad();
        check_broadcast_add_grad();
        std::cout << "gradcheck ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "gradcheck failed: " << ex.what() << std::endl;
        return 1;
    }
}
