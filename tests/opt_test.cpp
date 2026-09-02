#include <gtest/gtest.h>
#include "minijax/opt.hpp"
#include "minijax/egraph.hpp"
#include "minijax/interp.hpp"
#include "fixtures.hpp"

using namespace minijax;


TEST(Opt, LossFixtureOptimizedMatchesOriginal) {
    auto f = test_fixtures::make_loss_fixture();
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    Tensor before = eval(f.g, inputs, f.loss);
    auto [optimized, new_out] = optimize(f.g, f.loss);
    Tensor after = eval(optimized, inputs, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after, 1e-9, 1e-9));
}

TEST(Opt, AddZeroEliminated) {
    Graph g;
    NodeId a = g.input({3});
    NodeId zero = g.constant(0.0, {3});
    NodeId c = g.add(a, zero);
    NodeId loss = g.sum(c);

    Tensor input = Tensor::from_vec({3}, {1, 2, 3});
    Tensor before = eval(g, {input}, loss);

    auto [optimized, new_out] = optimize(g, loss);
    Tensor after = eval(optimized, {input}, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after));


    EXPECT_LT(optimized.size(), g.size());
}

TEST(Opt, MulOneAndMulZero) {
    Graph g;
    NodeId a = g.input({2});
    NodeId one = g.constant(1.0, {2});
    NodeId zero = g.constant(0.0, {2});
    NodeId m1 = g.mul(a, one);
    NodeId m2 = g.mul(a, zero);
    NodeId loss = g.sum(g.add(m1, m2));

    Tensor input = Tensor::from_vec({2}, {7, -3});
    Tensor before = eval(g, {input}, loss);
    auto [optimized, new_out] = optimize(g, loss);
    Tensor after = eval(optimized, {input}, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after));
}

TEST(Opt, DoubleNegationCancels) {
    Graph g;
    NodeId a = g.input({2});
    NodeId nn = g.neg(g.neg(a));
    NodeId loss = g.sum(nn);

    Tensor input = Tensor::from_vec({2}, {4, -6});
    Tensor before = eval(g, {input}, loss);
    auto [optimized, new_out] = optimize(g, loss);
    Tensor after = eval(optimized, {input}, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after));
    EXPECT_LT(optimized.size(), g.size());
}

TEST(Opt, AssociativityExposesHiddenZero) {


    Graph g;
    NodeId a = g.input({2});
    NodeId b = g.input({2});
    NodeId zero = g.constant(0.0, {2});
    NodeId chain = g.add(g.add(a, zero), b);
    NodeId loss = g.sum(chain);

    Tensor ta = Tensor::from_vec({2}, {1, 2}), tb = Tensor::from_vec({2}, {10, 20});
    Tensor before = eval(g, {ta, tb}, loss);
    auto [optimized, new_out] = optimize(g, loss);
    Tensor after = eval(optimized, {ta, tb}, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after));
}

TEST(Opt, ConstantFoldingArithmetic) {
    Graph g;
    NodeId c2 = g.constant(2.0, {});
    NodeId c3 = g.constant(3.0, {});
    NodeId sum = g.add(c2, c3);
    Tensor result = eval(g, {}, sum);
    EXPECT_DOUBLE_EQ(result.item(), 5.0);

    auto [optimized, new_out] = optimize(g, sum);
    Tensor after = eval(optimized, {}, new_out);
    EXPECT_DOUBLE_EQ(after.item(), 5.0);

    EXPECT_EQ(optimized.node(new_out).op, OpKind::Const);
}

TEST(Opt, MatmulAssociativityPreservesResult) {
    Graph g;
    NodeId p = g.input({2, 3});
    NodeId q = g.input({3, 4});
    NodeId b = g.input({4, 2});
    NodeId chain = g.matmul(g.matmul(p, q), b);
    NodeId loss = g.sum(chain);

    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6}),
        Tensor::from_vec({3, 4}, {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0}),
        Tensor::from_vec({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}),
    };
    Tensor before = eval(g, inputs, loss);
    auto [optimized, new_out] = optimize(g, loss);
    Tensor after = eval(optimized, inputs, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after, 1e-9, 1e-6));
}

TEST(Opt, InputOrderPreservedAfterOptimize) {
    auto f = test_fixtures::make_loss_fixture();
    auto [optimized, new_out] = optimize(f.g, f.loss);
    (void)new_out;
    ASSERT_EQ(optimized.num_inputs(), f.g.num_inputs());
    for (size_t i = 0; i < f.g.num_inputs(); ++i) {
        EXPECT_EQ(optimized.shape_of(optimized.inputs()[i]), f.g.shape_of(f.g.inputs()[i]));
    }
}

TEST(Opt, OptimizeSoundStillEliminatesAddZero) {


    Graph g;
    NodeId a = g.input({3});
    NodeId zero = g.constant(0.0, {3});
    NodeId c = g.add(a, zero);
    NodeId loss = g.sum(c);

    Tensor input = Tensor::from_vec({3}, {1, 2, 3});
    Tensor before = eval(g, {input}, loss);
    auto [optimized, new_out] = optimize_sound(g, loss);
    Tensor after = eval(optimized, {input}, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after));
    EXPECT_LT(optimized.size(), g.size());
}

TEST(Opt, OptimizeSoundStillCorrectOnMatmulChain) {


    Graph g;
    NodeId p = g.input({2, 3});
    NodeId q = g.input({3, 4});
    NodeId b = g.input({4, 2});
    NodeId chain = g.matmul(g.matmul(p, q), b);
    NodeId loss = g.sum(chain);

    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6}),
        Tensor::from_vec({3, 4}, {1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0}),
        Tensor::from_vec({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8}),
    };
    Tensor before = eval(g, inputs, loss);
    auto [optimized, new_out] = optimize_sound(g, loss);
    Tensor after = eval(optimized, inputs, new_out);
    EXPECT_TRUE(Tensor::allclose(before, after, 1e-9, 1e-6));
}


TEST(EGraph, HashconsDedupsIdenticalNodes) {
    EGraph eg;
    ENode a; a.op = OpKind::Const; a.const_value = 5.0; a.shape = {};
    EClassId id1 = eg.add(a);
    EClassId id2 = eg.add(a);
    EXPECT_EQ(id1, id2);
}

TEST(EGraph, UniteMergesClasses) {
    EGraph eg;
    ENode a; a.op = OpKind::Const; a.const_value = 1.0; a.shape = {};
    ENode b; b.op = OpKind::Const; b.const_value = 2.0; b.shape = {};
    EClassId id1 = eg.add(a);
    EClassId id2 = eg.add(b);
    EXPECT_NE(eg.find(id1), eg.find(id2));
    eg.unite(id1, id2);
    EXPECT_EQ(eg.find(id1), eg.find(id2));
    EXPECT_EQ(eg.members_of(id1).size(), 2u);
}

TEST(EGraph, SaturateTerminates) {
    Graph g;
    NodeId a = g.input({2});
    NodeId b = g.input({2});
    NodeId c = g.input({2});
    NodeId chain = g.add(g.add(a, b), c);
    auto [eg, node_to_eclass] = to_egraph(g);
    (void)node_to_eclass;
    int iters = saturate(eg, 8);
    EXPECT_LE(iters, 8);
}
