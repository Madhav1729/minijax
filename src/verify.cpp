#include "minijax/verify.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace minijax {

namespace {

std::vector<z3::expr> make_vars(z3::context& c, int arity, bool is_fpa) {
    static const char* names[] = {"a", "b", "c"};
    std::vector<z3::expr> vars;
    for (int i = 0; i < arity; ++i) {
        vars.push_back(is_fpa ? c.fpa_const(names[i], 11, 53) : c.real_const(names[i]));
    }
    return vars;
}

std::string model_to_string(z3::model& m, const std::vector<z3::expr>& vars) {
    static const char* names[] = {"a", "b", "c"};
    std::ostringstream os;
    for (size_t i = 0; i < vars.size(); ++i) {
        if (i) os << ", ";
        os << names[i] << "=" << m.eval(vars[i], true);
    }
    return os.str();
}


bool counterexample_is_nan_driven(z3::model& m, const std::vector<z3::expr>& vars,
                                   const z3::expr& lhs, const z3::expr& rhs) {
    for (const auto& v : vars) {
        std::string s = m.eval(v, true).to_string();
        if (s.find("NaN") != std::string::npos) return true;
    }

    if (m.eval(lhs.mk_is_nan(), true).is_true()) return true;
    if (m.eval(rhs.mk_is_nan(), true).is_true()) return true;
    return false;
}

}

RuleCheckResult Verifier::check_rule(const std::string& name, int arity, RuleExprFn lhs_fn, RuleExprFn rhs_fn,
                                      bool , bool ) {


    RuleCheckResult res;
    res.rule_name = name;


    {
        z3::context c;
        auto vars = make_vars(c, arity, false);
        z3::expr zero = c.real_val(0);
        z3::expr one = c.real_val(1);
        z3::expr lhs = lhs_fn(vars, zero, one);
        z3::expr rhs = rhs_fn(vars, zero, one);
        z3::solver s(c);
        s.add(lhs != rhs);
        z3::check_result r = s.check();
        res.sound_over_reals = (r == z3::unsat);
        if (r == z3::sat) {
            z3::model m = s.get_model();
            res.real_counterexample = model_to_string(m, vars);
        }
    }


    {
        z3::context c;
        auto vars = make_vars(c, arity, true);
        z3::expr zero = c.fpa_val(0.0);
        z3::expr one = c.fpa_val(1.0);
        z3::expr lhs = lhs_fn(vars, zero, one);
        z3::expr rhs = rhs_fn(vars, zero, one);
        z3::solver s(c);
        s.add(!z3::fp_eq(lhs, rhs));
        z3::check_result r = s.check();
        res.sound_over_floats = (r == z3::unsat);
        if (r == z3::sat) {
            z3::model m = s.get_model();
            std::string binding = model_to_string(m, vars);
            if (counterexample_is_nan_driven(m, vars, lhs, rhs)) {
                binding += "  [NaN-driven: the equality fails on NaN != NaN under fp.eq — "
                           "either an operand is NaN or the op produced NaN from a "
                           "signed-zero/Inf operand; not a rounding/Inf-driven counterexample]";
            } else if (binding.find("+oo") != std::string::npos || binding.find("-oo") != std::string::npos) {
                binding += "  [Inf-driven: an operand is infinite]";
            }
            res.float_counterexample = binding;
        }
    }

    if (res.sound_over_reals && res.sound_over_floats) res.classification = Soundness::SoundEverywhere;
    else if (res.sound_over_reals) res.classification = Soundness::RealsOnly;
    else res.classification = Soundness::Unsound;

    return res;
}

std::vector<RuleCheckResult> Verifier::soundness_report() {
    std::vector<RuleCheckResult> results;

    results.push_back(check_rule(
        "add_comm", 2,
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[0] + v[1]; },
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[1] + v[0]; }));

    results.push_back(check_rule(
        "mul_comm", 2,
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[0] * v[1]; },
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[1] * v[0]; }));

    results.push_back(check_rule(
        "add_assoc", 3,
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return (v[0] + v[1]) + v[2]; },
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[0] + (v[1] + v[2]); }));

    results.push_back(check_rule(
        "mul_assoc", 3,
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return (v[0] * v[1]) * v[2]; },
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[0] * (v[1] * v[2]); }));

    results.push_back(check_rule(
        "add_zero", 1,
        [](std::vector<z3::expr>& v, z3::expr zero, z3::expr) { return v[0] + zero; },
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[0]; }));

    results.push_back(check_rule(
        "mul_one", 1,
        [](std::vector<z3::expr>& v, z3::expr, z3::expr one) { return v[0] * one; },
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[0]; }));

    results.push_back(check_rule(
        "mul_zero", 1,
        [](std::vector<z3::expr>& v, z3::expr zero, z3::expr) { return v[0] * zero; },
        [](std::vector<z3::expr>&, z3::expr zero, z3::expr) { return zero; }));

    results.push_back(check_rule(
        "neg_neg", 1,
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return -(-v[0]); },
        [](std::vector<z3::expr>& v, z3::expr, z3::expr) { return v[0]; }));

    return results;
}

std::string format_soundness_report(const std::vector<RuleCheckResult>& results) {
    std::ostringstream os;
    os << std::left << std::setw(14) << "rule" << std::setw(14) << "sound(reals)"
       << std::setw(14) << "sound(float)" << "classification\n";
    os << std::string(14 + 14 + 14 + 16, '-') << "\n";
    for (const auto& r : results) {
        os << std::left << std::setw(14) << r.rule_name
           << std::setw(14) << (r.sound_over_reals ? "yes" : "NO")
           << std::setw(14) << (r.sound_over_floats ? "yes" : "NO");
        switch (r.classification) {
            case Soundness::SoundEverywhere: os << "sound-everywhere"; break;
            case Soundness::RealsOnly: os << "reals-only"; break;
            case Soundness::Unsound: os << "UNSOUND"; break;
        }
        os << "\n";
        if (r.float_counterexample && !r.sound_over_floats) {
            os << "    float counterexample: " << *r.float_counterexample << "\n";
        }
        if (r.real_counterexample && !r.sound_over_reals) {
            os << "    real counterexample: " << *r.real_counterexample << "\n";
        }
    }
    return os.str();
}

}
