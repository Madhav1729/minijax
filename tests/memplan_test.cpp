#include <gtest/gtest.h>
#include "minijax/memplan.hpp"
#include "minijax/vm.hpp"
#include "minijax/interp.hpp"

using namespace minijax;

TEST(MemPlan, SharedLivenessMatchesManualAnalysis) {


    Graph g;
    NodeId a = g.input({3});
    NodeId b = g.relu(a);
    NodeId c = g.sum(b);

    std::vector<StepAccess> steps = {
        {0, {}},
        {1, {0}},
        {2, {1}},
    };
    auto last_use = compute_last_use(3, steps);
    EXPECT_EQ(last_use[0], 1u);
    EXPECT_EQ(last_use[1], 2u);
    EXPECT_EQ(last_use[2], 0u);
    last_use[static_cast<size_t>(c)] = SIZE_MAX;
    EXPECT_EQ(last_use[2], SIZE_MAX);
}

TEST(MemPlan, DeepMlpPeakMemoryReductionGate) {


    constexpr size_t D = 16, LAYERS = 30;
    Graph g;
    NodeId cur = g.input({D, D});
    NodeId w = g.constant(0.5, {D, D});
    for (size_t i = 0; i < LAYERS; ++i) {
        cur = g.tanh(g.matmul(cur, w));
    }
    NodeId out = g.sum(cur);

    MemPlan plan = plan_memory(g, out);

    EXPECT_GT(plan.peak_bytes_unplanned, plan.peak_bytes_planned);


    EXPECT_GE(plan.reduction_ratio(), 0.5)
        << "unplanned=" << plan.peak_bytes_unplanned
        << " planned=" << plan.peak_bytes_planned;
}

TEST(MemPlan, SlotsReuseAcrossSameNumelShapes) {


    Graph g;
    NodeId v = g.input({4});
    NodeId m1 = g.relu(v);
    NodeId m2 = g.reshape(m1, {2, 2});
    NodeId m3 = g.abs(m2);
    NodeId out = g.sum(m3);

    MemPlan plan = plan_memory(g, out);
    EXPECT_LT(plan.num_slots, 5u);
    EXPECT_GT(plan.peak_bytes_unplanned, plan.peak_bytes_planned);
}

TEST(MemPlan, InPlaceCandidatesDetectedAndFiltered) {


    Graph g;
    NodeId a = g.input({3});
    NodeId b = g.relu(a);
    NodeId c = g.tanh(b);
    NodeId d = g.add(c, a);
    NodeId out = g.sum(d);

    MemPlan plan = plan_memory(g, out);
    auto has = [&](NodeId p, NodeId u) {
        for (const auto& ip : plan.in_place_candidates)
            if (ip.producer == p && ip.consumer == u) return true;
        return false;
    };
    EXPECT_TRUE(has(b, c));
    EXPECT_TRUE(has(c, d));
    EXPECT_FALSE(has(a, b)) << "Input-owned buffers must never be in-place targets";
    EXPECT_FALSE(has(a, d));
}

TEST(MemPlan, MultiConsumerBreaksInPlaceAndFusion) {
    Graph g;
    NodeId x = g.input({4});
    NodeId t = g.abs(x);
    NodeId u1 = g.sigmoid(t);
    NodeId u2 = g.neg(t);
    NodeId out = g.sum(g.add(u1, u2));

    MemPlan plan = plan_memory(g, out);
    for (const auto& ip : plan.in_place_candidates) {
        EXPECT_NE(ip.producer, t) << "multi-consumer node must not be an in-place target";
    }


    for (const auto& chain : plan.fusion_chains) {
        for (NodeId v : chain) EXPECT_NE(v, t);
    }
}

TEST(MemPlan, FusionChainsDetectedOnUnaryStack) {
    Graph g;
    NodeId x = g.input({4});
    NodeId y1 = g.tanh(x);
    NodeId y2 = g.tanh(y1);
    NodeId y3 = g.tanh(y2);
    NodeId out = g.sum(y3);

    MemPlan plan = plan_memory(g, out);
    ASSERT_EQ(plan.fusion_chains.size(), 1u);
    EXPECT_EQ(plan.fusion_chains[0], (std::vector<NodeId>{y1, y2, y3}));
}

TEST(MemPlan, ReportContainsSummarySections) {
    Graph g;
    NodeId x = g.input({4});
    NodeId y = g.sum(g.relu(x));
    MemPlan plan = plan_memory(g, y);
    std::string report = format_memory_report(g, plan);
    EXPECT_NE(report.find("peak memory (unplanned)"), std::string::npos);
    EXPECT_NE(report.find("peak memory (planned)"), std::string::npos);
    EXPECT_NE(report.find("in-place candidates"), std::string::npos);
    EXPECT_NE(report.find("fusion chains"), std::string::npos);
}
