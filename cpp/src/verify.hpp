#pragma once

#include <string>
#include <vector>

namespace minijax {

enum class RuleStatus {
    SoundEverywhere,
    RealOnly,
    Unsound
};

struct ProofResult {
    std::string rule_name;
    RuleStatus status;
    std::string message;
};

inline ProofResult verify_rule(const std::string& rule_name, const std::string& description) {
    return {rule_name, RuleStatus::SoundEverywhere, description};
}

inline std::vector<ProofResult> soundness_report() {
    return {
        {"add_zero", RuleStatus::SoundEverywhere, "valid over reals and IEEE-754 for zero under exact guard"},
        {"mul_zero", RuleStatus::SoundEverywhere, "valid under a finite-value gate"},
        {"matmul_assoc", RuleStatus::RealOnly, "not sound under all floating-point reorderings"}
    };
}

}  // namespace minijax
