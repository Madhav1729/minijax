#pragma once


#include <cstdint>
#include <vector>
#include <optional>
#include "minijax/ir.hpp"
#include "minijax/tensor.hpp"

namespace minijax {

enum class VMOp : uint8_t {
    LoadInput, LoadConst,
    Add, Sub, Mul, Div, Neg,
    MatMul,
    Relu, Step, Tanh, Sigmoid, Exp, Log, Sqrt, Abs,
    Sum, SumAxis,
    Transpose, Broadcast, Reshape,
};

struct Instr {
    VMOp op;
    uint32_t dst = 0;
    uint32_t src1 = 0;
    uint32_t src2 = 0;
    double const_value = 0.0;
    uint32_t input_slot = 0;
    std::vector<size_t> shape;
    size_t axis = 0;

    bool operator==(const Instr& o) const;
};

struct Program {
    std::vector<Instr> instrs;
    uint32_t num_regs = 0;
    uint32_t output_reg = 0;

    bool operator==(const Program& o) const;
};


Program compile(const Graph& g, NodeId output);


Program compact(const Program& p);


Tensor run_vm(const Program& p, const std::vector<Tensor>& inputs);


std::vector<uint8_t> serialize(const Program& p);
Program deserialize(const std::vector<uint8_t>& bytes);

}
