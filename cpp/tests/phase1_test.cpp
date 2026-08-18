#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "interp.hpp"
#include "ir.hpp"
#include "tensor.hpp"

namespace {

void expect_true(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

void expect_close(double actual, double expected, double tol = 1e-9, const std::string& msg = "") {
    if (std::abs(actual - expected) > tol) {
        throw std::runtime_error(msg.empty() ? "value mismatch" : msg);
    }
}

}  // namespace

int main() {
    try {
        // Core IR/build checks.
        minijax::Graph g;
        size_t w = g.input({2, 3});
        size_t x = g.input({3, 1});
        size_t wx = g.matmul(w, x);
        expect_true(g.nodes[wx].shape == std::vector<size_t>({2, 1}), "matmul shape mismatch");

        auto bshape = minijax::broadcast_shapes({2, 3}, {3});
        expect_true(bshape == std::vector<size_t>({2, 3}), "broadcast rule 1 failed");

        bshape = minijax::broadcast_shapes({2, 1}, {2, 3});
        expect_true(bshape == std::vector<size_t>({2, 3}), "broadcast rule 2 failed");

        bshape = minijax::broadcast_shapes({}, {4});
        expect_true(bshape == std::vector<size_t>({4}), "broadcast rule 3 failed");

        bshape = minijax::broadcast_shapes({5, 1, 4}, {3, 1});
        expect_true(bshape == std::vector<size_t>({5, 3, 4}), "broadcast rule 4 failed");

        bool threw = false;
        try {
            minijax::Graph bad;
            auto a = bad.input({2, 3});
            auto b = bad.input({4, 1});
            bad.matmul(a, b);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect_true(threw, "matmul mismatch should throw");

        // Interpreter checks.
        minijax::Graph addg;
        auto a = addg.input({});
        auto b = addg.input({});
        auto c = addg.add(a, b);
        auto values = minijax::eval(addg, {minijax::Tensor::scalar(3.0), minijax::Tensor::scalar(4.0)});
        expect_close(values[c].scalar(), 7.0, 1e-9, "scalar add failed");

        minijax::Graph mm;
        auto m1 = mm.input({2, 2});
        auto m2 = mm.input({2, 2});
        auto prod = mm.matmul(m1, m2);
        minijax::Tensor av = minijax::Tensor::from_matrix({{1.0, 2.0}, {3.0, 4.0}});
        minijax::Tensor bv = minijax::Tensor::from_matrix({{5.0, 6.0}, {7.0, 8.0}});
        auto mv = minijax::eval(mm, {av, bv});
        expect_close(mv[prod].at({0, 0}), 19.0, 1e-9, "matmul 0,0 failed");
        expect_close(mv[prod].at({1, 1}), 50.0, 1e-9, "matmul 1,1 failed");

        minijax::Graph relu_g;
        auto in = relu_g.input({4});
        auto r = relu_g.relu(in);
        minijax::Tensor vec = minijax::Tensor({4}, {-2.0, -1.0, 0.0, 3.0});
        auto relu_vals = minijax::eval(relu_g, {vec});
        expect_close(relu_vals[r].at_1d(0), 0.0, 1e-9, "relu clamp failed");
        expect_close(relu_vals[r].at_1d(3), 3.0, 1e-9, "relu value failed");

        minijax::Graph sumg;
        auto sv = sumg.input({3});
        auto s = sumg.sum(sv);
        minijax::Tensor svec = minijax::Tensor({3}, {1.0, 2.0, 3.0});
        auto sum_vals = minijax::eval(sumg, {svec});
        expect_close(sum_vals[s].scalar(), 6.0, 1e-9, "sum reduction failed");

        minijax::Graph biasg;
        auto a2 = biasg.input({2, 3});
        auto b2 = biasg.input({3});
        auto c2 = biasg.add(a2, b2);
        minijax::Tensor av2 = minijax::Tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
        minijax::Tensor bv2 = minijax::Tensor({3}, {10.0, 20.0, 30.0});
        auto bias_vals = minijax::eval(biasg, {av2, bv2});
        expect_true(bias_vals[c2].shape == std::vector<size_t>({2, 3}), "bias broadcast shape wrong");
        expect_close(bias_vals[c2].at({0, 0}), 11.0, 1e-9, "bias row 0 failed");
        expect_close(bias_vals[c2].at({1, 2}), 36.0, 1e-9, "bias row 1 failed");

        minijax::Graph colg;
        auto a3 = colg.input({2, 1});
        auto b3 = colg.input({2, 3});
        auto c3 = colg.mul(a3, b3);
        minijax::Tensor av3 = minijax::Tensor({2, 1}, {2.0, 3.0});
        minijax::Tensor bv3 = minijax::Tensor({2, 3}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
        auto col_vals = minijax::eval(colg, {av3, bv3});
        expect_close(col_vals[c3].at({0, 1}), 2.0, 1e-9, "column broadcast failed");
        expect_close(col_vals[c3].at({1, 2}), 3.0, 1e-9, "column broadcast failed");

        minijax::Graph lossg;
        auto W = lossg.input({2, 2});
        auto X = lossg.input({2, 1});
        auto Y = lossg.input({2, 1});
        auto WX = lossg.matmul(W, X);
        auto act = lossg.relu(WX);
        auto diff = lossg.sub(act, Y);
        auto sq = lossg.mul(diff, diff);
        auto loss = lossg.sum(sq);
        minijax::Tensor Wv = minijax::Tensor::from_matrix({{1.0, 0.0}, {0.0, 1.0}});
        minijax::Tensor Xv = minijax::Tensor::from_matrix({{2.0}, {3.0}});
        minijax::Tensor Yv = minijax::Tensor::from_matrix({{1.0}, {1.0}});
        auto loss_vals = minijax::eval(lossg, {Wv, Xv, Yv});
        expect_close(loss_vals[loss].scalar(), 5.0, 1e-9, "loss example failed");

        std::cout << "phase1 ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "phase1 failed: " << ex.what() << std::endl;
        return 1;
    }
}
