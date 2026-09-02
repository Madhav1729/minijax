#include <gtest/gtest.h>
#include "minijax/autodiff.hpp"
#include "minijax/interp.hpp"
#include "fixtures.hpp"
#include "gradcheck.hpp"

using namespace minijax;
using minijax::test_fixtures::check_gradient;

TEST(Autodiff, AddGradIsOnes) {
    Graph g;
    NodeId a = g.input({2});
    NodeId b = g.input({2});
    NodeId c = g.add(a, b);
    NodeId loss = g.sum(c);
    auto grads = grad(g, loss, {a, b});
    Tensor ga = eval(g, {Tensor::from_vec({2}, {1, 2}), Tensor::from_vec({2}, {3, 4})}, grads[0]);
    Tensor gb = eval(g, {Tensor::from_vec({2}, {1, 2}), Tensor::from_vec({2}, {3, 4})}, grads[1]);
    EXPECT_TRUE(Tensor::allclose(ga, Tensor::from_vec({2}, {1, 1})));
    EXPECT_TRUE(Tensor::allclose(gb, Tensor::from_vec({2}, {1, 1})));
}

TEST(Autodiff, MulGradIsOtherOperand) {
    Graph g;
    NodeId a = g.input({2});
    NodeId b = g.input({2});
    NodeId c = g.mul(a, b);
    NodeId loss = g.sum(c);
    check_gradient(g, loss, a, {Tensor::from_vec({2}, {3, -2}), Tensor::from_vec({2}, {5, 7})});
}

TEST(Autodiff, MatMulGradcheck) {
    Graph g;
    NodeId a = g.input({2, 3});
    NodeId b = g.input({3, 2});
    NodeId c = g.matmul(a, b);
    NodeId loss = g.sum(c);
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6}),
        Tensor::from_vec({3, 2}, {0.5, -1, 2, 0.1, -0.3, 4}),
    };
    check_gradient(g, loss, a, inputs);
    check_gradient(g, loss, b, inputs);
}

TEST(Autodiff, ReluGradcheck) {
    Graph g;
    NodeId a = g.input({4});
    NodeId r = g.relu(a);
    NodeId loss = g.sum(r);

    check_gradient(g, loss, a, {Tensor::from_vec({4}, {-3, -0.5, 0.7, 2})});
}

TEST(Autodiff, TanhGradcheck) {
    Graph g;
    NodeId a = g.input({3});
    NodeId t = g.tanh(a);
    NodeId loss = g.sum(t);
    check_gradient(g, loss, a, {Tensor::from_vec({3}, {-1.0, 0.3, 2.0})});
}

TEST(Autodiff, SigmoidGradcheck) {
    Graph g;
    NodeId a = g.input({3});
    NodeId s = g.sigmoid(a);
    NodeId loss = g.sum(s);
    check_gradient(g, loss, a, {Tensor::from_vec({3}, {-1.0, 0.3, 2.0})});
}

TEST(Autodiff, ExpLogGradcheck) {
    Graph g;
    NodeId a = g.input({3});
    NodeId e = g.exp(a);
    NodeId l = g.log(e);
    NodeId loss = g.sum(l);
    check_gradient(g, loss, a, {Tensor::from_vec({3}, {0.5, 1.2, -0.3})});
}

TEST(Autodiff, SqrtGradcheck) {
    Graph g;
    NodeId a = g.input({3});
    NodeId s = g.sqrt(a);
    NodeId loss = g.sum(s);
    check_gradient(g, loss, a, {Tensor::from_vec({3}, {4.0, 9.0, 2.25})});
}

TEST(Autodiff, DivGradcheck) {
    Graph g;
    NodeId a = g.input({3});
    NodeId b = g.input({3});
    NodeId d = g.div(a, b);
    NodeId loss = g.sum(d);
    std::vector<Tensor> inputs = {Tensor::from_vec({3}, {1, 2, 3}), Tensor::from_vec({3}, {2, 4, 5})};
    check_gradient(g, loss, a, inputs);
    check_gradient(g, loss, b, inputs);
}

TEST(Autodiff, SumAxisGradcheck) {
    Graph g;
    NodeId a = g.input({2, 3});
    NodeId s = g.sum_axis(a, 1);
    NodeId loss = g.sum(s);
    check_gradient(g, loss, a, {Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6})});
}

TEST(Autodiff, BroadcastGradSumsOverExpandedAxes) {
    Graph g;
    NodeId a = g.input({3});
    NodeId b = g.input({2, 3});
    NodeId c = g.add(a, b);
    NodeId loss = g.sum(c);
    check_gradient(g, loss, a, {Tensor::from_vec({3}, {1, 2, 3}), Tensor::from_vec({2, 3}, {1, 1, 1, 1, 1, 1})});
}

TEST(Autodiff, LossFixtureGradcheckAllInputs) {
    auto f = test_fixtures::make_loss_fixture();
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    check_gradient(f.g, f.loss, f.W, inputs);
    check_gradient(f.g, f.loss, f.x, inputs);
    check_gradient(f.g, f.loss, f.y, inputs);
}

TEST(Autodiff, NonScalarOutputThrows) {
    Graph g;
    NodeId a = g.input({3});
    EXPECT_THROW(grad(g, a, {a}), std::invalid_argument);
}

TEST(Autodiff, UnusedWrtGivesZeroGrad) {
    Graph g;
    NodeId a = g.input({2});
    NodeId unused = g.input({2});
    NodeId loss = g.sum(a);
    auto grads = grad(g, loss, {unused});
    Tensor gz = eval(g, {Tensor::from_vec({2}, {1, 1}), Tensor::from_vec({2}, {9, 9})}, grads[0]);
    EXPECT_TRUE(Tensor::allclose(gz, Tensor::zeros({2})));
}
