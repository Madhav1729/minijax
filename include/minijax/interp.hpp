#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include "minijax/ir.hpp"
#include "minijax/tensor.hpp"

namespace minijax {


struct OpProfile {
    std::unordered_map<OpKind, double> ns_per_op;
    double total_ns = 0.0;

    const char* unit() const { return "ns"; }
};


Tensor eval(const Graph& g, const std::vector<Tensor>& inputs, NodeId output);


std::vector<Tensor> eval_all(const Graph& g, const std::vector<Tensor>& inputs,
                              OpProfile* profile = nullptr);

}
