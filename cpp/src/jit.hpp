#pragma once

#include <stdexcept>

#include "ir.hpp"
#include "tensor.hpp"

namespace minijax {

class JitBackend {
public:
    Tensor execute(const Graph& graph, const std::vector<Tensor>& inputs) const {
        (void)graph;
        (void)inputs;
        throw std::runtime_error("JIT backend not implemented yet");
    }
};

}  // namespace minijax
