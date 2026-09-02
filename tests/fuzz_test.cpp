#include <gtest/gtest.h>
#include "minijax/fuzz.hpp"
#include "minijax/interp.hpp"
#include "fixtures.hpp"

using namespace minijax;

TEST(Fuzz, GeneratedProgramIsWellFormedAndEvaluable) {
    auto prog = generate_random_program(1, 8, 4);
    EXPECT_GT(prog.g.size(), 0u);
    EXPECT_EQ(prog.inputs.size(), prog.g.num_inputs());


    Tensor result = eval(prog.g, prog.inputs, prog.output);
    EXPECT_GE(result.rank(), 0u);
}

TEST(Fuzz, GenerationIsDeterministicGivenSameSeed) {
    auto a = generate_random_program(7, 6, 3);
    auto b = generate_random_program(7, 6, 3);
    EXPECT_EQ(a.g.size(), b.g.size());
    EXPECT_EQ(a.inputs.size(), b.inputs.size());
    for (size_t i = 0; i < a.inputs.size(); ++i) {
        EXPECT_TRUE(Tensor::allclose(a.inputs[i], b.inputs[i], 0.0, 0.0));
    }
}

TEST(Fuzz, InterpVsVmOracleAgreesOnKnownGoodGraph) {
    Graph g;
    NodeId a = g.input({2, 2});
    NodeId b = g.input({2, 2});
    NodeId c = g.matmul(a, b);
    NodeId loss = g.sum(c);
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {1, 2, 3, 4}),
        Tensor::from_vec({2, 2}, {5, 6, 7, 8}),
    };
    auto r = check_interp_vs_vm(g, loss, inputs);
    EXPECT_TRUE(r.passed) << r.detail;
}

TEST(Fuzz, MetamorphicOracleAgreesOnKnownGoodGraph) {
    Graph g;
    NodeId a = g.input({3});
    NodeId zero = g.constant(0.0, {3});
    NodeId c = g.add(a, zero);
    NodeId loss = g.sum(c);
    auto r = check_metamorphic_optimize_sound(g, loss, {Tensor::from_vec({3}, {1, 2, 3})});
    EXPECT_TRUE(r.passed) << r.detail;
}


TEST(Fuzz, SeededCampaignFindsNoFailures) {
    auto result = run_fuzz_campaign(42, 300, 8, 4);
    EXPECT_EQ(result.iterations_run, 300);
    std::string report;
    for (const auto& f : result.failures) {
        report += "[" + f.oracle + "] seed=" + std::to_string(f.seed) +
                  " depth=" + std::to_string(f.depth) + ": " + f.detail + "\n";
    }
    EXPECT_TRUE(result.failures.empty()) << report;
}

#if defined(MINIJAX_FUZZ_HAS_JIT_ORACLE)
TEST(Fuzz, InterpVsJitOracleAgreesOnKnownGoodGraph) {
    auto f = test_fixtures::make_loss_fixture();
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    OracleResult r = check_interp_vs_jit(f.g, f.loss, inputs);
    EXPECT_TRUE(r.passed) << r.detail;
}


TEST(Fuzz, SeededCampaignWithJitOracleFindsNoFailures) {
    auto result = run_fuzz_campaign(7, 100,
                                     8, 4,
                                     true);
    EXPECT_EQ(result.iterations_run, 100);
    std::string report;
    for (const auto& f : result.failures) {
        report += "[" + f.oracle + "] seed=" + std::to_string(f.seed) +
                  " depth=" + std::to_string(f.depth) + ": " + f.detail + "\n";
    }
    EXPECT_TRUE(result.failures.empty()) << report;
}
#endif
