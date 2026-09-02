#pragma once


#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <z3++.h>

namespace minijax {

enum class Soundness {
    SoundEverywhere,
    RealsOnly,
    Unsound,
};

struct RuleCheckResult {
    std::string rule_name;
    bool sound_over_reals = false;
    bool sound_over_floats = false;
    Soundness classification = Soundness::Unsound;
    std::optional<std::string> float_counterexample;
    std::optional<std::string> real_counterexample;
};


using RuleExprFn = std::function<z3::expr(std::vector<z3::expr>& vars, z3::expr zero, z3::expr one)>;

class Verifier {
public:


    RuleCheckResult check_rule(const std::string& name, int arity, RuleExprFn lhs, RuleExprFn rhs,
                                bool exclude_nan = true, bool exclude_inf = false);


    std::vector<RuleCheckResult> soundness_report();
};


std::string format_soundness_report(const std::vector<RuleCheckResult>& results);

}

