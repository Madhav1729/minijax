#include <gtest/gtest.h>
#include "minijax/interp.hpp"
#include "fixtures.hpp"

using namespace minijax;

TEST(Interp, LossFixtureHandComputed) {


    auto f = test_fixtures::make_loss_fixture();
    Tensor W = Tensor::from_vec({2, 2}, {1, 0, 0, 1});
    Tensor x = Tensor::from_vec({2, 1}, {2, 3});
    Tensor y = Tensor::from_vec({2, 1}, {1, 1});
    Tensor result = eval(f.g, {W, x, y}, f.loss);
    EXPECT_DOUBLE_EQ(result.item(), 3.0);
}

TEST(Interp, ReluZerosOutNegatives) {
    Graph g;
    NodeId a = g.input({3});
    NodeId r = g.relu(a);
    Tensor result = eval(g, {Tensor::from_vec({3}, {-1, 0, 5})}, r);
    Tensor expected = Tensor::from_vec({3}, {0, 0, 5});
    EXPECT_TRUE(Tensor::allclose(result, expected));
}

TEST(Interp, ConstNodeBroadcastsToItsShape) {
    Graph g;
    NodeId c = g.constant(2.5, {2, 2});
    Tensor result = eval(g, {}, c);
    EXPECT_TRUE(Tensor::allclose(result, Tensor::full({2, 2}, 2.5)));
}

TEST(Interp, WrongNumberOfInputsThrows) {
    auto f = test_fixtures::make_loss_fixture();
    EXPECT_THROW(eval(f.g, {Tensor::scalar(1.0)}, f.loss), std::invalid_argument);
}

TEST(Interp, ExplicitBroadcastNodeMatchesTensorBroadcast) {
    Graph g;
    NodeId a = g.input({2, 3});
    NodeId b = g.input({3});
    NodeId c = g.add(a, b);
    Tensor ta = Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor tb = Tensor::from_vec({3}, {10, 20, 30});
    Tensor result = eval(g, {ta, tb}, c);
    Tensor expected = Tensor::from_vec({2, 3}, {11, 22, 33, 14, 25, 36});
    EXPECT_TRUE(Tensor::allclose(result, expected));
}

TEST(Interp, EvalAllReturnsPerNodeValues) {
    Graph g;
    NodeId a = g.input({1});
    NodeId b = g.constant(3.0, {1});
    NodeId c = g.mul(a, b);
    auto values = eval_all(g, {Tensor::from_vec({1}, {4.0})});
    ASSERT_EQ(values.size(), g.size());
    EXPECT_DOUBLE_EQ(values[c].data()[0], 12.0);
}
