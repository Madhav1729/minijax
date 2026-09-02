#include "minijax/autodiff.hpp"
#include <stdexcept>

namespace minijax {

namespace {

void accumulate(Graph& g, std::vector<NodeId>& adj, NodeId target, NodeId contrib) {
    if (adj[target] == kInvalidNode) {
        adj[target] = contrib;
    } else {
        adj[target] = g.add(adj[target], contrib);
    }
}


NodeId sum_to(Graph& g, NodeId x, const std::vector<size_t>& target_shape) {
    if (target_shape.empty()) {
        return g.sum(x);
    }
    NodeId cur = x;
    size_t cur_rank = g.shape_of(cur).size();
    size_t extra = cur_rank - target_shape.size();
    for (size_t i = 0; i < extra; ++i) {
        cur = g.sum_axis(cur, 0);
    }
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == 1 && g.shape_of(cur)[i] != 1) {
            NodeId summed = g.sum_axis(cur, i);
            std::vector<size_t> restored = g.shape_of(summed);
            restored.insert(restored.begin() + static_cast<long>(i), 1);
            cur = g.reshape(summed, restored);
        }
    }
    return cur;
}

}

std::vector<NodeId> grad(Graph& g, NodeId output, const std::vector<NodeId>& wrt) {
    if (!g.shape_of(output).empty()) {
        throw std::invalid_argument("grad: output must be scalar (rank-0)");
    }

    const size_t initial_size = g.size();
    std::vector<NodeId> adj(initial_size, kInvalidNode);
    adj[output] = g.constant(1.0, {});

    for (NodeId id = initial_size; id-- > 0;) {
        if (adj[id] == kInvalidNode) continue;
        NodeId gy = adj[id];
        Node node = g.node(id);


        switch (node.op) {
            case OpKind::Input:
            case OpKind::Const:
                break;

            case OpKind::Add:
                accumulate(g, adj, node.inputs[0], gy);
                accumulate(g, adj, node.inputs[1], gy);
                break;

            case OpKind::Sub:
                accumulate(g, adj, node.inputs[0], gy);
                accumulate(g, adj, node.inputs[1], g.neg(gy));
                break;

            case OpKind::Mul:
                accumulate(g, adj, node.inputs[0], g.mul(gy, node.inputs[1]));
                accumulate(g, adj, node.inputs[1], g.mul(gy, node.inputs[0]));
                break;

            case OpKind::Div: {

                NodeId a = node.inputs[0], b = node.inputs[1];
                accumulate(g, adj, a, g.div(gy, b));
                NodeId a_over_b2 = g.div(g.mul(gy, a), g.mul(b, b));
                accumulate(g, adj, b, g.neg(a_over_b2));
                break;
            }

            case OpKind::Neg:
                accumulate(g, adj, node.inputs[0], g.neg(gy));
                break;

            case OpKind::MatMul: {

                NodeId a = node.inputs[0], b = node.inputs[1];
                accumulate(g, adj, a, g.matmul(gy, g.transpose(b)));
                accumulate(g, adj, b, g.matmul(g.transpose(a), gy));
                break;
            }

            case OpKind::Relu: {

                NodeId mask = g.step(node.inputs[0]);
                accumulate(g, adj, node.inputs[0], g.mul(gy, mask));
                break;
            }

            case OpKind::Step:
                break;

            case OpKind::Tanh: {

                NodeId one = g.constant(1.0, node.shape);
                NodeId out_sq = g.mul(id, id);
                NodeId deriv = g.sub(one, out_sq);
                accumulate(g, adj, node.inputs[0], g.mul(gy, deriv));
                break;
            }

            case OpKind::Sigmoid: {

                NodeId one = g.constant(1.0, node.shape);
                NodeId one_minus = g.sub(one, id);
                NodeId deriv = g.mul(id, one_minus);
                accumulate(g, adj, node.inputs[0], g.mul(gy, deriv));
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

            case OpKind::Sqrt: {

                NodeId two = g.constant(2.0, node.shape);
                accumulate(g, adj, node.inputs[0], g.div(gy, g.mul(two, id)));
                break;
            }

            case OpKind::Abs: {

                NodeId two = g.constant(2.0, node.shape);
                NodeId one = g.constant(1.0, node.shape);
                NodeId sign = g.sub(g.mul(two, g.step(node.inputs[0])), one);
                accumulate(g, adj, node.inputs[0], g.mul(gy, sign));
                break;
            }

            case OpKind::Sum:
                accumulate(g, adj, node.inputs[0], g.broadcast_to(gy, g.shape_of(node.inputs[0])));
                break;

            case OpKind::SumAxis: {

                std::vector<size_t> expanded = g.shape_of(gy);
                expanded.insert(expanded.begin() + static_cast<long>(node.axis), 1);
                NodeId reshaped = g.reshape(gy, expanded);
                accumulate(g, adj, node.inputs[0],
                           g.broadcast_to(reshaped, g.shape_of(node.inputs[0])));
                break;
            }

            case OpKind::Transpose:
                accumulate(g, adj, node.inputs[0], g.transpose(gy));
                break;

            case OpKind::Broadcast:
                accumulate(g, adj, node.inputs[0], sum_to(g, gy, g.shape_of(node.inputs[0])));
                break;

            case OpKind::Reshape:
                accumulate(g, adj, node.inputs[0], g.reshape(gy, g.shape_of(node.inputs[0])));
                break;

            case OpKind::Im2Col: {
                const auto& x_shape = g.shape_of(node.inputs[0]);
                accumulate(g, adj, node.inputs[0],
                           g.col2im(gy, x_shape[0], x_shape[1], x_shape[2], node.kernel_h, node.kernel_w,
                                    node.stride, node.pad));
                break;
            }

            case OpKind::Col2Im: {


                accumulate(g, adj, node.inputs[0],
                           g.im2col(gy, node.kernel_h, node.kernel_w, node.stride, node.pad));
                break;
            }

            case OpKind::MaxPool: {


                accumulate(g, adj, node.inputs[0],
                           g.maxpool_backward(node.inputs[0], id, gy, node.kernel_h, node.kernel_w, node.stride));
                break;
            }

            case OpKind::MaxPoolBackward:


                throw std::runtime_error("grad: MaxPoolBackward is a backward-only op; "
                                          "second-order autodiff is not supported");

            case OpKind::Softmax:
            case OpKind::CrossEntropy:
            case OpKind::Conv2d:
            case OpKind::BatchNorm:
                throw std::runtime_error(std::string("grad: op is composite-only, never directly emitted (Phase 9): ") +
                                          op_kind_name(node.op));
        }
    }

    std::vector<NodeId> result;
    result.reserve(wrt.size());
    for (NodeId w : wrt) {
        if (w >= initial_size || adj[w] == kInvalidNode) {

            result.push_back(g.constant(0.0, g.shape_of(w)));
        } else {
            result.push_back(adj[w]);
        }
    }
    return result;
}

}
