#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

namespace minijax {

enum class OpKind {
    Input,
    Const,
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Relu,
    Step,
    Exp,
    Log,
    Tanh,
    Sigmoid,
    MatMul,
    Transpose,
    Broadcast,
    Reshape,
    Sum,
    SumAxis
};

struct Node {
    OpKind op;
    std::vector<size_t> inputs;
    Shape shape;
    double const_value = 0.0;
    size_t axis = 0;
    size_t slot = 0;
};

class Graph {
public:
    std::vector<Node> nodes;
    std::vector<size_t> inputs;

    size_t push(OpKind op, std::vector<size_t> in, Shape out_shape, double const_value = 0.0, size_t axis = 0) {
        nodes.push_back(Node{op, std::move(in), std::move(out_shape), const_value, axis});
        return nodes.size() - 1;
    }

    size_t input(const Shape& shape) {
        size_t idx = inputs.size();
        size_t id = push(OpKind::Input, {}, shape);
        nodes[id].slot = idx;
        inputs.push_back(id);
        return id;
    }

    size_t constant(double value, const Shape& shape) {
        return push(OpKind::Const, {}, shape, value);
    }

    size_t add(size_t a, size_t b) { return binop(OpKind::Add, a, b); }
    size_t sub(size_t a, size_t b) { return binop(OpKind::Sub, a, b); }
    size_t mul(size_t a, size_t b) { return binop(OpKind::Mul, a, b); }
    size_t div(size_t a, size_t b) { return binop(OpKind::Div, a, b); }

    size_t neg(size_t a) {
        return push(OpKind::Neg, {a}, nodes[a].shape);
    }

    size_t relu(size_t a) {
        return push(OpKind::Relu, {a}, nodes[a].shape);
    }

    size_t step(size_t a) {
        return push(OpKind::Step, {a}, nodes[a].shape);
    }

    size_t exp(size_t a) {
        return push(OpKind::Exp, {a}, nodes[a].shape);
    }

    size_t log(size_t a) {
        return push(OpKind::Log, {a}, nodes[a].shape);
    }

    size_t tanh(size_t a) {
        return push(OpKind::Tanh, {a}, nodes[a].shape);
    }

    size_t sigmoid(size_t a) {
        return push(OpKind::Sigmoid, {a}, nodes[a].shape);
    }

    size_t matmul(size_t a, size_t b) {
        const auto& sa = nodes[a].shape;
        const auto& sb = nodes[b].shape;
        if (sa.size() != 2 || sb.size() != 2) {
            throw std::runtime_error("matmul expects 2D inputs");
        }
        if (sa[1] != sb[0]) {
            throw std::runtime_error("matmul inner-dim mismatch");
        }
        Shape out = {sa[0], sb[1]};
        return push(OpKind::MatMul, {a, b}, out);
    }

    size_t transpose(size_t a) {
        const auto& s = nodes[a].shape;
        if (s.size() != 2) {
            throw std::runtime_error("transpose expects 2D input");
        }
        return push(OpKind::Transpose, {a}, {s[1], s[0]});
    }

    size_t broadcast(size_t a, const Shape& target) {
        return push(OpKind::Broadcast, {a}, target);
    }

    size_t broadcast_to(size_t a, const Shape& target) {
        if (nodes[a].shape == target) {
            return a;
        }
        return broadcast(a, target);
    }

    size_t reshape(size_t a, const Shape& target) {
        return push(OpKind::Reshape, {a}, target);
    }

    size_t sum(size_t a) {
        return push(OpKind::Sum, {a}, {});
    }

    size_t sum_axis(size_t a, size_t axis) {
        Shape out = nodes[a].shape;
        if (axis >= out.size()) {
            throw std::runtime_error("sum_axis axis out of range");
        }
        out.erase(out.begin() + static_cast<std::ptrdiff_t>(axis));
        size_t id = push(OpKind::SumAxis, {a}, out, 0.0, axis);
        nodes[id].axis = axis;
        return id;
    }

    size_t num_inputs() const { return inputs.size(); }

private:
    size_t binop(OpKind op, size_t a, size_t b) {
        Shape out = broadcast_shapes(nodes[a].shape, nodes[b].shape);
        size_t aa = broadcast_to(a, out);
        size_t bb = broadcast_to(b, out);
        return push(op, {aa, bb}, out);
    }
};

}  // namespace minijax
