#include <gtest/gtest.h>
#include "minijax/ir.hpp"
#include "fixtures.hpp"

using namespace minijax;

TEST(Ir, TopoOrderInvariant) {
    Graph g;
    NodeId a = g.input({2});
    NodeId b = g.input({2});
    NodeId c = g.add(a, b);

    for (NodeId in : g.node(c).inputs) EXPECT_LT(in, c);
}

TEST(Ir, InputSlotsTrackOrder) {
    Graph g;
    NodeId a = g.input({1});
    NodeId b = g.input({1});
    ASSERT_EQ(g.num_inputs(), 2u);
    EXPECT_EQ(g.inputs()[0], a);
    EXPECT_EQ(g.inputs()[1], b);
    EXPECT_EQ(g.node(a).input_slot, 0u);
    EXPECT_EQ(g.node(b).input_slot, 1u);
}

TEST(Ir, MatMulShapeInference) {
    Graph g;
    NodeId a = g.input({3, 4});
    NodeId b = g.input({4, 5});
    NodeId c = g.matmul(a, b);
    EXPECT_EQ(g.shape_of(c), (std::vector<size_t>{3, 5}));
}

TEST(Ir, MatMulShapeMismatchThrows) {
    Graph g;
    NodeId a = g.input({3, 4});
    NodeId b = g.input({6, 5});
    EXPECT_THROW(g.matmul(a, b), std::invalid_argument);
}

TEST(Ir, BinopInsertsExplicitBroadcastNode) {
    Graph g;
    NodeId a = g.input({2, 3});
    NodeId b = g.input({3});
    NodeId c = g.add(a, b);


    NodeId second_input = g.node(c).inputs[1];
    EXPECT_NE(second_input, b);
    EXPECT_EQ(g.node(second_input).op, OpKind::Broadcast);
    EXPECT_EQ(g.shape_of(c), (std::vector<size_t>{2, 3}));
}

TEST(Ir, IncompatibleBroadcastThrows) {
    Graph g;
    NodeId a = g.input({2, 3});
    NodeId b = g.input({4});
    EXPECT_THROW(g.add(a, b), std::invalid_argument);
}

TEST(Ir, SumReducesToRank0) {
    Graph g;
    NodeId a = g.input({2, 2});
    NodeId s = g.sum(a);
    EXPECT_EQ(g.shape_of(s).size(), 0u);
}

TEST(Ir, LossFixtureBuildsAndShapesOut) {
    auto f = test_fixtures::make_loss_fixture();
    EXPECT_EQ(f.g.shape_of(f.loss).size(), 0u);
    EXPECT_GT(f.g.size(), 0u);
}
