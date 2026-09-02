#pragma once


#include <memory>
#include <stdexcept>
#include <vector>
#include "minijax/ir.hpp"
#include "minijax/tensor.hpp"

namespace minijax {


extern "C" void mj_matmul(const double* a, const double* b, double* out,
                           size_t m, size_t k, size_t n);

namespace jit {


class JitProgram {
public:


    explicit JitProgram(const Graph& g);
    ~JitProgram();
    JitProgram(JitProgram&&) noexcept;
    JitProgram& operator=(JitProgram&&) noexcept;
    JitProgram(const JitProgram&) = delete;
    JitProgram& operator=(const JitProgram&) = delete;


    void execute(const std::vector<Tensor>& inputs);


    Tensor value(NodeId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;


    size_t num_nodes_ = 0;
    std::vector<std::vector<size_t>> shapes_;
    std::vector<NodeId> input_slots_;


    std::vector<std::vector<double>> bufs_;
    std::vector<double*> buf_ptrs_;
    std::vector<double*> input_ptrs_;
};


Tensor run_jit(const Graph& g, const std::vector<Tensor>& inputs, NodeId output);

}
}
