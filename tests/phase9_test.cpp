#include <gtest/gtest.h>
#include <cmath>
#include "minijax/ir.hpp"
#include "minijax/interp.hpp"
#include "minijax/autodiff.hpp"
#include "gradcheck.hpp"

using namespace minijax;
using minijax::test_fixtures::check_gradient;


TEST(Phase9, Im2ColMatchesTensorDirectly) {
    Graph g;
    NodeId x = g.input({1, 3, 3});
    NodeId cols = g.im2col(x, 2, 2, 1, 0);
    Tensor xt = Tensor::from_vec({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor expected = xt.im2col(2, 2, 1, 0);
    Tensor got = eval(g, {xt}, cols);
    EXPECT_TRUE(Tensor::allclose(got, expected));
}

TEST(Phase9, MaxPoolMatchesTensorDirectly) {
    Graph g;
    NodeId x = g.input({1, 4, 4});
    NodeId pooled = g.maxpool(x, 2, 2, 2);
    Tensor xt = Tensor::from_vec({1, 4, 4}, {1, 3, 2, 4, 5, 6, 1, 2, 9, 8, 7, 0, 1, 2, 3, 4});
    Tensor expected = xt.maxpool(2, 2, 2);
    Tensor got = eval(g, {xt}, pooled);
    EXPECT_TRUE(Tensor::allclose(got, expected));
}


TEST(Phase9, Im2ColGradcheck) {
    Graph g;
    NodeId x = g.input({1, 3, 3});
    NodeId cols = g.im2col(x, 2, 2, 1, 0);
    NodeId loss = g.sum(cols);
    check_gradient(g, loss, x, {Tensor::from_vec({1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9})});
}

TEST(Phase9, MaxPoolGradcheck) {
    Graph g;
    NodeId x = g.input({1, 4, 4});
    NodeId pooled = g.maxpool(x, 2, 2, 2);
    NodeId loss = g.sum(pooled);


    check_gradient(g, loss, x,
                    {Tensor::from_vec({1, 4, 4}, {1.1, 3.2, 2.1, 4.3, 5.5, 6.6, 1.2, 2.4, 9.1, 8.2, 7.3, 0.4, 1.5, 2.6, 3.7, 4.8})});
}

TEST(Phase9, Conv2dGradcheck) {
    Graph g;
    NodeId x = g.input({1, 4, 4});
    NodeId kernel = g.input({2, 1, 2, 2});
    NodeId out = g.conv2d(x, kernel, 1, 0);
    NodeId loss = g.sum(out);

    std::vector<Tensor> inputs = {
        Tensor::from_vec({1, 4, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}),
        Tensor::from_vec({2, 1, 2, 2}, {0.1, -0.2, 0.3, 0.4, -0.1, 0.2, -0.3, 0.5}),
    };
    check_gradient(g, loss, x, inputs);
    check_gradient(g, loss, kernel, inputs);
}

TEST(Phase9, Conv2dOutputShapeCorrect) {
    Graph g;
    NodeId x = g.input({3, 8, 8});
    NodeId kernel = g.input({5, 3, 3, 3});
    NodeId out = g.conv2d(x, kernel, 2, 1);

    EXPECT_EQ(g.shape_of(out), (std::vector<size_t>{5, 4, 4}));
}

TEST(Phase9, SoftmaxSumsToOne) {
    Graph g;
    NodeId x = g.input({4});
    NodeId s = g.softmax(x);
    Tensor result = eval(g, {Tensor::from_vec({4}, {1.0, 2.0, -1.0, 0.5})}, s);
    EXPECT_NEAR(result.sum().item(), 1.0, 1e-9);
    for (double v : result.data()) EXPECT_GT(v, 0.0);
}

TEST(Phase9, SoftmaxGradcheck) {
    Graph g;
    NodeId x = g.input({4});
    NodeId s = g.softmax(x);


    NodeId weights = g.input({4});
    NodeId loss = g.sum(g.mul(s, weights));
    check_gradient(g, loss, x,
                    {Tensor::from_vec({4}, {1.0, 2.0, -1.0, 0.5}), Tensor::from_vec({4}, {0.3, -0.7, 1.1, 0.2})});
}

TEST(Phase9, CrossEntropyMatchesHandComputed) {
    Graph g;
    NodeId pred = g.input({3});
    NodeId target = g.input({3});
    NodeId loss = g.cross_entropy(pred, target);

    Tensor p = Tensor::from_vec({3}, {0.2, 0.5, 0.3});
    Tensor t = Tensor::from_vec({3}, {0.0, 1.0, 0.0});
    Tensor result = eval(g, {p, t}, loss);
    EXPECT_NEAR(result.item(), -std::log(0.5), 1e-9);
}

TEST(Phase9, SoftmaxCrossEntropyGradcheck) {
    Graph g;
    NodeId logits = g.input({3});
    NodeId target = g.input({3});
    NodeId probs = g.softmax(logits);
    NodeId loss = g.cross_entropy(probs, target);
    check_gradient(g, loss, logits,
                    {Tensor::from_vec({3}, {0.5, -0.3, 1.2}), Tensor::from_vec({3}, {0.0, 1.0, 0.0})}, 1e-4, 1e-3);
}

TEST(Phase9, BatchNormNormalizesToZeroMeanUnitVar) {
    Graph g;
    NodeId x = g.input({4});
    NodeId gamma = g.input({4});
    NodeId beta = g.input({4});
    NodeId out = g.batch_norm(x, gamma, beta, 1e-8);
    Tensor result = eval(g, {Tensor::from_vec({4}, {1, 2, 3, 4}), Tensor::full({4}, 1.0), Tensor::full({4}, 0.0)}, out);

    EXPECT_NEAR(result.sum().item(), 0.0, 1e-6);
}

TEST(Phase9, BatchNormGradcheck) {
    Graph g;
    NodeId x = g.input({4});
    NodeId gamma = g.input({4});
    NodeId beta = g.input({4});
    NodeId out = g.batch_norm(x, gamma, beta, 1e-4);
    NodeId loss = g.sum(out);
    std::vector<Tensor> inputs = {
        Tensor::from_vec({4}, {1.0, 2.5, -1.0, 0.3}),
        Tensor::from_vec({4}, {1.2, 0.8, 1.1, 0.9}),
        Tensor::from_vec({4}, {0.1, -0.1, 0.2, 0.0}),
    };
    check_gradient(g, loss, x, inputs);
    check_gradient(g, loss, gamma, inputs);
    check_gradient(g, loss, beta, inputs);
}
