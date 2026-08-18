#pragma once

#include <optional>
#include <vector>

#include "ir.hpp"

namespace minijax {

inline size_t sum_to(Graph& g, size_t node_id, const Shape& target) {
    Shape cur = g.nodes[node_id].shape;
    size_t extra = cur.size() > target.size() ? cur.size() - target.size() : 0;
    for (size_t i = 0; i < extra; ++i) {
        node_id = g.sum_axis(node_id, 0);
    }
    Shape cur2 = g.nodes[node_id].shape;
    for (size_t i = 0; i < target.size(); ++i) {
        if (target[i] == 1 && cur2[i] != 1) {
            node_id = g.sum_axis(node_id, i);
            Shape s = g.nodes[node_id].shape;
            s.insert(s.begin() + static_cast<std::ptrdiff_t>(i), 1);
            node_id = g.reshape(node_id, s);
        }
    }
    return node_id;
}

inline void accumulate(Graph& g, std::vector<std::optional<size_t>>& adj, size_t id, size_t contrib) {
    if (id >= adj.size()) adj.resize(id + 1);
    if (adj[id].has_value()) {
        size_t prev = adj[id].value();
        adj[id] = g.add(prev, contrib);
    } else {
        adj[id] = contrib;
    }
}

inline std::vector<size_t> grad(Graph& g, size_t output, const std::vector<size_t>& wrt) {
    size_t n = g.nodes.size();
    std::vector<std::optional<size_t>> adj(n);

    size_t one = g.constant(1.0, g.nodes[output].shape);
    adj[output] = one;

    for (int id = static_cast<int>(n) - 1; id >= 0; --id) {
        if (!adj[id].has_value()) {
            continue;
        }
        size_t gy = adj[id].value();
        Node node = g.nodes[id];
        auto op_name = [](OpKind k) -> const char* {
            switch (k) {
                case OpKind::Add: return "Add";
                case OpKind::Sub: return "Sub";
                case OpKind::Mul: return "Mul";
                case OpKind::Div: return "Div";
                case OpKind::Neg: return "Neg";
                case OpKind::Relu: return "Relu";
                case OpKind::Exp: return "Exp";
                case OpKind::Log: return "Log";
                case OpKind::Tanh: return "Tanh";
                case OpKind::Sigmoid: return "Sigmoid";
                case OpKind::MatMul: return "MatMul";
                case OpKind::Transpose: return "Transpose";
                case OpKind::Sum: return "Sum";
                case OpKind::SumAxis: return "SumAxis";
                case OpKind::Broadcast: return "Broadcast";
                case OpKind::Reshape: return "Reshape";
                case OpKind::Input: return "Input";
                case OpKind::Const: return "Const";
                case OpKind::Step: return "Step";
                default: return "Unknown";
            }
        };

        // Defensive checks to catch invalid input indexing
        if ((node.op == OpKind::Add || node.op == OpKind::Sub || node.op == OpKind::Mul || node.op == OpKind::Div || node.op == OpKind::MatMul) && node.inputs.size() < 2) {
            throw std::runtime_error(std::string("node has insufficient inputs for op ") + op_name(node.op) + " id=" + std::to_string(id));
        }
        switch (node.op) {
            case OpKind::Add: {
                accumulate(g, adj, node.inputs[0], gy);
                accumulate(g, adj, node.inputs[1], gy);
                break;
            }
            case OpKind::Sub: {
                accumulate(g, adj, node.inputs[0], gy);
                size_t neg_gy = g.neg(gy);
                accumulate(g, adj, node.inputs[1], neg_gy);
                break;
            }
            case OpKind::Mul: {
                size_t a = node.inputs[0];
                size_t b = node.inputs[1];
                accumulate(g, adj, a, g.mul(gy, b));
                accumulate(g, adj, b, g.mul(gy, a));
                break;
            }
            case OpKind::Div: {
                size_t a = node.inputs[0];
                size_t b = node.inputs[1];
                accumulate(g, adj, a, g.div(gy, b));
                size_t b2 = g.mul(b, b);
                size_t neg_a = g.neg(a);
                size_t tmp = g.div(neg_a, b2);
                size_t gb = g.mul(gy, tmp);
                accumulate(g, adj, b, gb);
                break;
            }
            case OpKind::Neg: {
                accumulate(g, adj, node.inputs[0], g.neg(gy));
                break;
            }
            case OpKind::Relu: {
                size_t mask = g.step(node.inputs[0]);
                accumulate(g, adj, node.inputs[0], g.mul(gy, mask));
                break;
            }
            case OpKind::Exp: {
                accumulate(g, adj, node.inputs[0], g.mul(gy, id));
                break;
            }
            case OpKind::Log: {
                accumulate(g, adj, node.inputs[0], g.div(gy, node.inputs[0]));
                break;
            }
            case OpKind::Tanh: {
                size_t tanh_out = id;
                size_t tanh_sq = g.mul(tanh_out, tanh_out);
                size_t one = g.constant(1.0, g.nodes[id].shape);
                size_t dtanh = g.sub(one, tanh_sq);
                accumulate(g, adj, node.inputs[0], g.mul(gy, dtanh));
                break;
            }
            case OpKind::Sigmoid: {
                size_t one = g.constant(1.0, g.nodes[id].shape);
                size_t one_minus = g.sub(one, id);
                size_t ds = g.mul(id, one_minus);
                accumulate(g, adj, node.inputs[0], g.mul(gy, ds));
                break;
            }
            case OpKind::MatMul: {
                size_t a = node.inputs[0];
                size_t b = node.inputs[1];
                size_t bt = g.transpose(b);
                accumulate(g, adj, a, g.matmul(gy, bt));
                size_t at = g.transpose(a);
                accumulate(g, adj, b, g.matmul(at, gy));
                break;
            }
            case OpKind::Transpose: {
                accumulate(g, adj, node.inputs[0], g.transpose(gy));
                break;
            }
            case OpKind::Sum: {
                accumulate(g, adj, node.inputs[0], g.broadcast(gy, g.nodes[node.inputs[0]].shape));
                break;
            }
            case OpKind::SumAxis: {
                Shape in_shape = g.nodes[node.inputs[0]].shape;
                Shape exp_shape = g.nodes[gy].shape;
                exp_shape.insert(exp_shape.begin() + static_cast<std::ptrdiff_t>(node.axis), 1);
                size_t expanded = g.reshape(gy, exp_shape);
                accumulate(g, adj, node.inputs[0], g.broadcast(expanded, in_shape));
                break;
            }
            case OpKind::Broadcast: {
                accumulate(g, adj, node.inputs[0], sum_to(g, gy, g.nodes[node.inputs[0]].shape));
                break;
            }
            case OpKind::Reshape: {
                accumulate(g, adj, node.inputs[0], g.reshape(gy, g.nodes[node.inputs[0]].shape));
                break;
            }
            case OpKind::Input:
            case OpKind::Const:
            case OpKind::Step:
                break;
            default:
                throw std::runtime_error("unsupported gradient op");
        }
    }

    std::vector<size_t> result;
    result.reserve(wrt.size());
    for (size_t w : wrt) {
        if (!adj[w].has_value()) {
            throw std::runtime_error("no gradient path to requested input");
        }
        result.push_back(adj[w].value());
    }
    return result;
}

}  // namespace minijax
