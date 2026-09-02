#include <gtest/gtest.h>
#include "minijax/verify.hpp"

using namespace minijax;

TEST(Verify, CommutativityIsSoundOverReals) {
    Verifier v;
    auto r = v.check_rule("add_comm", 2,
        [](std::vector<z3::expr>& vars, z3::expr, z3::expr) { return vars[0] + vars[1]; },
        [](std::vector<z3::expr>& vars, z3::expr, z3::expr) { return vars[1] + vars[0]; });
    EXPECT_TRUE(r.sound_over_reals);
}

TEST(Verify, AssociativityIsSoundOverRealsButNotFloats) {


    Verifier v;
    auto r = v.check_rule("add_assoc", 3,
        [](std::vector<z3::expr>& vars, z3::expr, z3::expr) { return (vars[0] + vars[1]) + vars[2]; },
        [](std::vector<z3::expr>& vars, z3::expr, z3::expr) { return vars[0] + (vars[1] + vars[2]); });
    EXPECT_TRUE(r.sound_over_reals);
    EXPECT_FALSE(r.sound_over_floats);
    EXPECT_EQ(r.classification, Soundness::RealsOnly);
    ASSERT_TRUE(r.float_counterexample.has_value());


    EXPECT_EQ(r.float_counterexample->find("NaN-driven"), std::string::npos);
}

TEST(Verify, MulZeroFlaggedUnsoundDueToInfNan) {
    Verifier v;
    auto r = v.check_rule("mul_zero", 1,
        [](std::vector<z3::expr>& vars, z3::expr zero, z3::expr) { return vars[0] * zero; },
        [](std::vector<z3::expr>&, z3::expr zero, z3::expr) { return zero; });
    EXPECT_TRUE(r.sound_over_reals);
    EXPECT_FALSE(r.sound_over_floats);
    ASSERT_TRUE(r.float_counterexample.has_value());
}

TEST(Verify, DeliberatelyWrongRuleIsCaughtEvenOverReals) {


    Verifier v;
    auto r = v.check_rule("deliberately_wrong", 2,
        [](std::vector<z3::expr>& vars, z3::expr, z3::expr) { return vars[0] + vars[1]; },
        [](std::vector<z3::expr>& vars, z3::expr, z3::expr) { return vars[0] - vars[1]; });
    EXPECT_FALSE(r.sound_over_reals);
    EXPECT_FALSE(r.sound_over_floats);
    EXPECT_EQ(r.classification, Soundness::Unsound);
    ASSERT_TRUE(r.real_counterexample.has_value());
}

TEST(Verify, SoundnessReportCoversAllOptimizerRules) {
    Verifier v;
    auto results = v.soundness_report();
    ASSERT_EQ(results.size(), 8u);


    for (const auto& r : results) {
        EXPECT_TRUE(r.sound_over_reals) << r.rule_name << " should be sound over reals";
    }


    auto find_result = [&](const std::string& name) -> const RuleCheckResult& {
        for (const auto& r : results) if (r.rule_name == name) return r;
        throw std::runtime_error("rule not found: " + name);
    };
    EXPECT_FALSE(find_result("add_assoc").sound_over_floats);
    EXPECT_FALSE(find_result("mul_assoc").sound_over_floats);
    EXPECT_FALSE(find_result("mul_zero").sound_over_floats);


    for (const char* name : {"add_comm", "mul_comm", "add_zero", "mul_one", "neg_neg"}) {
        const auto& r = find_result(name);
        if (!r.sound_over_floats) {
            ASSERT_TRUE(r.float_counterexample.has_value());
            EXPECT_NE(r.float_counterexample->find("NaN-driven"), std::string::npos)
                << name << "'s float-unsoundness should be NaN-driven, not a genuine rounding/Inf issue";
        }
    }
}

TEST(Verify, FormatSoundnessReportProducesNonEmptyTable) {
    Verifier v;
    auto results = v.soundness_report();
    std::string report = format_soundness_report(results);
    EXPECT_NE(report.find("add_assoc"), std::string::npos);
    EXPECT_NE(report.find("reals-only"), std::string::npos);
}
