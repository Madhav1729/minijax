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

void expect_true(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

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

void check_grad_for_loss() {
    minijax::Graph g;
    auto w = g.input({2, 2});
    auto x = g.input({2, 1});
    auto y = g.input({2, 1});
    auto wx = g.matmul(w, x);
    auto act = g.relu(wx);
    auto diff = g.sub(act, y);
    auto sq = g.mul(diff, diff);
    auto loss = g.sum(sq);

    minijax::Tensor W = minijax::Tensor::from_matrix({{1.0, 0.0}, {0.0, 1.0}});
    minijax::Tensor X = minijax::Tensor::from_matrix({{2.0}, {3.0}});
    minijax::Tensor Y = minijax::Tensor::from_matrix({{1.0}, {1.0}});

    auto grad_ids = minijax::grad(g, loss, {w, x, y});
    auto val = minijax::eval(g, {W, X, Y});

    // Grad wrt W should be [[4,6],[8,12]] for this example.
    auto grad_w = val[grad_ids[0]];
    expect_close(grad_w.at({0, 0}), 4.0, 1e-6, "grad_w[0,0]");
    expect_close(grad_w.at({0, 1}), 6.0, 1e-6, "grad_w[0,1]");
    expect_close(grad_w.at({1, 0}), 8.0, 1e-6, "grad_w[1,0]");
    expect_close(grad_w.at({1, 1}), 12.0, 1e-6, "grad_w[1,1]");

    // Numerical gradient checks (finite differences)
    {
        std::vector<minijax::Tensor> inputs = {W, X, Y};
        for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
            const auto& t = inputs[in_idx];
            for (size_t e = 0; e < t.numel(); ++e) {
                double fd = finite_diff_scalar(g, inputs, in_idx, e, loss);
                auto grad_tensor = val[grad_ids[in_idx]];
                double gval = grad_tensor.at_1d(e);
                expect_close(gval, fd, 1e-4, "numeric grad mismatch (loss)");
            }
        }
    }
}

void check_grad_for_relu() {
    minijax::Graph g;
    auto x = g.input({2});
    auto r = g.relu(x);
    auto s = g.sum(r);

    minijax::Tensor p = minijax::Tensor({2}, {1.5, -2.0});
    auto grads = minijax::grad(g, s, {x});
    auto vals = minijax::eval(g, {p});
    auto grad = vals[grads[0]];
    expect_close(grad.at_1d(0), 1.0, 1e-6, "relu grad x0");
    expect_close(grad.at_1d(1), 0.0, 1e-6, "relu grad x1");

    // Numerical gradient checks
    {
        std::vector<minijax::Tensor> inputs = {p};
        for (size_t e = 0; e < p.numel(); ++e) {
            double fd = finite_diff_scalar(g, inputs, 0, e, s);
            double gval = grad.at_1d(e);
            expect_close(gval, fd, 1e-6, "numeric grad mismatch (relu)");
        }
    }
}

void check_grad_for_addition() {
    minijax::Graph g;
    auto a = g.input({2});
    auto b = g.input({2});
    auto c = g.add(a, b);
    auto s = g.sum(c);

    minijax::Tensor pa = minijax::Tensor({2}, {1.0, 2.0});
    minijax::Tensor pb = minijax::Tensor({2}, {3.0, 4.0});
    auto grads = minijax::grad(g, s, {a, b});
    auto vals = minijax::eval(g, {pa, pb});
    auto ga = vals[grads[0]];
    auto gb = vals[grads[1]];

    expect_close(ga.at_1d(0), 1.0, 1e-6, "add grad a0");
    expect_close(ga.at_1d(1), 1.0, 1e-6, "add grad a1");
    expect_close(gb.at_1d(0), 1.0, 1e-6, "add grad b0");
    expect_close(gb.at_1d(1), 1.0, 1e-6, "add grad b1");

    // Numerical gradient checks
    {
        std::vector<minijax::Tensor> inputs = {pa, pb};
        for (size_t in_idx = 0; in_idx < inputs.size(); ++in_idx) {
            const auto& t = inputs[in_idx];
            for (size_t e = 0; e < t.numel(); ++e) {
                double fd = finite_diff_scalar(g, inputs, in_idx, e, s);
                auto grad_tensor = vals[grads[in_idx]];
                double gval = grad_tensor.at_1d(e);
                expect_close(gval, fd, 1e-6, "numeric grad mismatch (add)");
            }
        }
    }
}

}  // namespace

int main() {
    try {
        check_grad_for_loss();
        check_grad_for_relu();
        check_grad_for_addition();
        std::cout << "phase2 ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "phase2 failed: " << ex.what() << std::endl;
        return 1;
    }
}
