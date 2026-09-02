#include "minijax/interp.hpp"
#include <stdexcept>

namespace minijax {

std::vector<Tensor> eval_all(const Graph& g, const std::vector<Tensor>& inputs, OpProfile* profile) {
    if (inputs.size() != g.num_inputs()) {
        throw std::invalid_argument("eval: expected " + std::to_string(g.num_inputs()) +
                                     " inputs, got " + std::to_string(inputs.size()));
    }
    std::vector<Tensor> values;
    values.reserve(g.size());
    using clock = std::chrono::steady_clock;

    for (NodeId i = 0; i < g.size(); ++i) {
        const Node& n = g.node(i);
        auto t0 = profile ? clock::now() : clock::time_point{};
        switch (n.op) {
            case OpKind::Input:
                values.push_back(inputs[n.input_slot]);
                break;
            case OpKind::Const:
                values.push_back(Tensor::full(n.shape, n.const_value));
                break;
            case OpKind::Add:
                values.push_back(values[n.inputs[0]] + values[n.inputs[1]]);
                break;
            case OpKind::Sub:
                values.push_back(values[n.inputs[0]] - values[n.inputs[1]]);
                break;
            case OpKind::Mul:
                values.push_back(values[n.inputs[0]] * values[n.inputs[1]]);
                break;
            case OpKind::Div:
                values.push_back(values[n.inputs[0]] / values[n.inputs[1]]);
                break;
            case OpKind::Neg:
                values.push_back(-values[n.inputs[0]]);
                break;
            case OpKind::MatMul:
                values.push_back(Tensor::matmul(values[n.inputs[0]], values[n.inputs[1]]));
                break;
            case OpKind::Relu:
                values.push_back(values[n.inputs[0]].relu());
                break;
            case OpKind::Step:
                values.push_back(values[n.inputs[0]].step());
                break;
            case OpKind::Tanh:
                values.push_back(values[n.inputs[0]].tanh());
                break;
            case OpKind::Sigmoid:
                values.push_back(values[n.inputs[0]].sigmoid());
                break;
            case OpKind::Exp:
                values.push_back(values[n.inputs[0]].exp());
                break;
            case OpKind::Log:
                values.push_back(values[n.inputs[0]].log());
                break;
            case OpKind::Sqrt:
                values.push_back(values[n.inputs[0]].sqrt());
                break;
            case OpKind::Abs:
                values.push_back(values[n.inputs[0]].abs());
                break;
            case OpKind::Sum:
                values.push_back(values[n.inputs[0]].sum());
                break;
            case OpKind::SumAxis:
                values.push_back(values[n.inputs[0]].sum_axis(n.axis));
                break;
            case OpKind::Transpose:
                values.push_back(values[n.inputs[0]].transpose());
                break;
            case OpKind::Broadcast:
                values.push_back(values[n.inputs[0]].broadcast_to(n.target_shape));
                break;
            case OpKind::Reshape:
                values.push_back(values[n.inputs[0]].reshape(n.target_shape));
                break;
            case OpKind::Im2Col:
                values.push_back(values[n.inputs[0]].im2col(n.kernel_h, n.kernel_w, n.stride, n.pad));
                break;
            case OpKind::Col2Im:
                values.push_back(Tensor::col2im(values[n.inputs[0]], n.shape[0], n.shape[1], n.shape[2],
                                                 n.kernel_h, n.kernel_w, n.stride, n.pad));
                break;
            case OpKind::MaxPool:
                values.push_back(values[n.inputs[0]].maxpool(n.kernel_h, n.kernel_w, n.stride));
                break;
            case OpKind::MaxPoolBackward:
                values.push_back(Tensor::maxpool_backward(values[n.inputs[0]], values[n.inputs[1]],
                                                            values[n.inputs[2]], n.kernel_h, n.kernel_w, n.stride));
                break;
            case OpKind::Softmax:
            case OpKind::CrossEntropy:
            case OpKind::Conv2d:
            case OpKind::BatchNorm:
                throw std::runtime_error(std::string("eval: op is composite-only, never directly emitted (Phase 9): ") +
                                          op_kind_name(n.op));
        }
        if (profile) {
            auto t1 = clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            profile->ns_per_op[n.op] += ns;
            profile->total_ns += ns;
        }
    }
    return values;
}

Tensor eval(const Graph& g, const std::vector<Tensor>& inputs, NodeId output) {


    auto values = eval_all(g, inputs);
    return values[output];
}

}
