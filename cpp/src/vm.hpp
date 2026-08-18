#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ir.hpp"
#include "tensor.hpp"

namespace minijax {

enum class VMOp {
    LoadInput,
    Const,
    Unary,
    Binary,
    MatMul,
    Transpose,
    Broadcast,
    Reshape,
    Sum,
    SumAxis
};

struct VMInstr {
    VMOp op;
    size_t dst = 0;
    size_t a = 0;
    size_t b = 0;
    size_t index = 0;
    double value = 0.0;
    Shape shape;
    size_t axis = 0;
};

struct VMProgram {
    std::vector<VMInstr> instrs;
    size_t num_regs = 0;
    size_t output = 0;
    size_t num_inputs = 0;
};

inline VMProgram compile_graph(const Graph& g, size_t output_node) {
    VMProgram p;
    p.num_regs = g.nodes.size();
    p.output = output_node;
    p.num_inputs = g.inputs.size();

    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& node = g.nodes[i];
        VMInstr inst;
        inst.op = VMOp::Const;
        inst.dst = i;
        switch (node.op) {
            case OpKind::Input:
                inst.op = VMOp::LoadInput;
                inst.index = node.slot;
                break;
            case OpKind::Const:
                inst.value = node.const_value;
                inst.shape = node.shape;
                break;
            case OpKind::Add:
                inst.op = VMOp::Binary;
                inst.a = node.inputs[0];
                inst.b = node.inputs[1];
                break;
            case OpKind::Mul:
                inst.op = VMOp::Binary;
                inst.a = node.inputs[0];
                inst.b = node.inputs[1];
                break;
            case OpKind::MatMul:
                inst.op = VMOp::MatMul;
                inst.a = node.inputs[0];
                inst.b = node.inputs[1];
                break;
            case OpKind::Broadcast:
                inst.op = VMOp::Broadcast;
                inst.a = node.inputs[0];
                inst.shape = node.shape;
                break;
            case OpKind::Reshape:
                inst.op = VMOp::Reshape;
                inst.a = node.inputs[0];
                inst.shape = node.shape;
                break;
            case OpKind::Sum:
                inst.op = VMOp::Sum;
                inst.a = node.inputs[0];
                break;
            case OpKind::SumAxis:
                inst.op = VMOp::SumAxis;
                inst.a = node.inputs[0];
                inst.axis = node.axis;
                break;
            default:
                inst.op = VMOp::Unary;
                inst.a = node.inputs[0];
                break;
        }
        p.instrs.push_back(inst);
    }
    return p;
}

inline Tensor run_vm(const VMProgram& p, const std::vector<Tensor>& inputs) {
    std::vector<std::optional<Tensor>> regs(p.num_regs);
    for (const auto& instr : p.instrs) {
        switch (instr.op) {
            case VMOp::LoadInput:
                regs[instr.dst] = inputs[instr.index];
                break;
            case VMOp::Const:
                regs[instr.dst] = Tensor(instr.shape, std::vector<double>(product(instr.shape), instr.value));
                break;
            case VMOp::Binary: {
                auto a = regs[instr.a].value();
                auto b = regs[instr.b].value();
                regs[instr.dst] = a + b;
                break;
            }
            case VMOp::MatMul: {
                auto a = regs[instr.a].value();
                auto b = regs[instr.b].value();
                regs[instr.dst] = a.matmul(b);
                break;
            }
            case VMOp::Transpose: {
                regs[instr.dst] = regs[instr.a].value().transpose();
                break;
            }
            case VMOp::Broadcast: {
                regs[instr.dst] = regs[instr.a].value().broadcast_to(instr.shape);
                break;
            }
            case VMOp::Reshape: {
                regs[instr.dst] = regs[instr.a].value().reshape(instr.shape);
                break;
            }
            case VMOp::Sum: {
                regs[instr.dst] = regs[instr.a].value().sum();
                break;
            }
            case VMOp::SumAxis: {
                regs[instr.dst] = regs[instr.a].value().sum_axis(instr.axis);
                break;
            }
            case VMOp::Unary:
                throw std::runtime_error("unary VM ops not yet implemented");
        }
    }
    return regs[p.output].value();
}

}  // namespace minijax
