#pragma once

#include <functional>
#include <string>
#include <vector>

#include "ir.hpp"
#include "tensor.hpp"

namespace minijax {

struct RewriteRule {
    std::string name;
    std::function<Tensor(const Tensor&)> apply;
};

class Optimizer {
public:
    std::vector<RewriteRule> rules() const {
        return {
            {"add_zero", [](const Tensor& t) { return t; }},
            {"mul_one", [](const Tensor& t) { return t; }},
            {"mul_zero", [](const Tensor& t) { return Tensor::zeros(t.shape); }},
            {"commutative_add", [](const Tensor& t) { return t; }}
        };
    }

    Graph optimize(const Graph& graph) const {
        (void)graph;
        return graph;
    }

    Tensor optimize_tensor(const Tensor& t) const {
        Tensor out = t;
        for (const auto& rule : rules()) {
            out = rule.apply(out);
        }
        return out;
    }
};

}  // namespace minijax
