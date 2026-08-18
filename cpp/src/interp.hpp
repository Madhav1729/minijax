#pragma once

#include <optional>
#include <vector>

#include "ir.hpp"

namespace minijax {

inline Tensor get_value(const std::vector<std::optional<Tensor>>& values, size_t id) {
    if (!values[id].has_value()) {
        throw std::runtime_error("dependency not evaluated");
    }
    return values[id].value();
}

inline std::vector<Tensor> eval(const Graph& g, const std::vector<Tensor>& inputs) {
    if (inputs.size() != g.inputs.size()) {
        throw std::runtime_error("wrong number of inputs");
    }

    std::vector<std::optional<Tensor>> values(g.nodes.size());
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& node = g.nodes[i];
        Tensor v;
        switch (node.op) {
            case OpKind::Input:
                v = inputs[node.slot];
                break;
            case OpKind::Const:
                v = Tensor(node.shape, std::vector<double>(product(node.shape), node.const_value));
                break;
            case OpKind::Add:
                v = get_value(values, node.inputs[0]) + get_value(values, node.inputs[1]);
                break;
            case OpKind::Sub:
                v = get_value(values, node.inputs[0]) - get_value(values, node.inputs[1]);
                break;
            case OpKind::Mul:
                v = get_value(values, node.inputs[0]) * get_value(values, node.inputs[1]);
                break;
            case OpKind::Div:
                v = get_value(values, node.inputs[0]) / get_value(values, node.inputs[1]);
                break;
            case OpKind::Neg:
                v = -get_value(values, node.inputs[0]);
                break;
            case OpKind::Relu:
                v = get_value(values, node.inputs[0]).relu();
                break;
            case OpKind::Step:
                v = get_value(values, node.inputs[0]).step();
                break;
            case OpKind::Exp:
                v = get_value(values, node.inputs[0]).exp();
                break;
            case OpKind::Log:
                v = get_value(values, node.inputs[0]).log();
                break;
            case OpKind::Tanh:
                v = get_value(values, node.inputs[0]).tanh();
                break;
            case OpKind::Sigmoid:
                v = get_value(values, node.inputs[0]).sigmoid();
                break;
            case OpKind::MatMul:
                v = get_value(values, node.inputs[0]).matmul(get_value(values, node.inputs[1]));
                break;
            case OpKind::Transpose:
                v = get_value(values, node.inputs[0]).transpose();
                break;
            case OpKind::Broadcast:
                v = get_value(values, node.inputs[0]).broadcast_to(node.shape);
                break;
            case OpKind::Reshape:
                v = get_value(values, node.inputs[0]).reshape(node.shape);
                break;
            case OpKind::Sum:
                v = get_value(values, node.inputs[0]).sum();
                break;
            case OpKind::SumAxis:
                v = get_value(values, node.inputs[0]).sum_axis(node.axis);
                break;
            default:
                throw std::runtime_error("unsupported op");
        }
        values[i] = v;
    }

    std::vector<Tensor> out;
    out.reserve(g.nodes.size());
    for (const auto& value : values) {
        if (!value.has_value()) {
            throw std::runtime_error("node not evaluated");
        }
        out.push_back(value.value());
    }
    return out;
}

}  // namespace minijax
