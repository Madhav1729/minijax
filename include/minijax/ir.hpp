#pragma once


#include <vector>
#include <cstddef>
#include <string>
#include <variant>
#include <optional>
#include "minijax/tensor.hpp"

namespace minijax {

using NodeId = size_t;
constexpr NodeId kInvalidNode = static_cast<NodeId>(-1);

enum class OpKind {
    Input, Const,
    Add, Sub, Mul, Div, Neg,
    MatMul,
    Relu, Step, Tanh, Sigmoid, Exp, Log, Sqrt, Abs,
    Sum, SumAxis,
    Transpose, Broadcast, Reshape,
    Softmax, CrossEntropy,
    Conv2d, BatchNorm,
    Im2Col, Col2Im, MaxPool, MaxPoolBackward,
};

const char* op_kind_name(OpKind k);

struct Node {
    OpKind op;
    std::vector<NodeId> inputs;
    std::vector<size_t> shape;


    double const_value = 0.0;
    size_t input_slot = 0;
    size_t axis = 0;
    std::vector<size_t> target_shape;

    size_t stride = 1;
    size_t pad = 0;
    size_t kernel_h = 0, kernel_w = 0;
};

class Graph {
public:

    NodeId input(std::vector<size_t> shape);
    NodeId constant(double value, std::vector<size_t> shape = {});

    NodeId add(NodeId a, NodeId b);
    NodeId sub(NodeId a, NodeId b);
    NodeId mul(NodeId a, NodeId b);
    NodeId div(NodeId a, NodeId b);
    NodeId neg(NodeId a);

    NodeId matmul(NodeId a, NodeId b);
    NodeId relu(NodeId a);
    NodeId step(NodeId a);
    NodeId tanh(NodeId a);
    NodeId sigmoid(NodeId a);
    NodeId exp(NodeId a);
    NodeId log(NodeId a);
    NodeId sqrt(NodeId a);
    NodeId abs(NodeId a);

    NodeId sum(NodeId a);
    NodeId sum_axis(NodeId a, size_t axis);
    NodeId transpose(NodeId a);
    NodeId broadcast_to(NodeId a, std::vector<size_t> shape);
    NodeId reshape(NodeId a, std::vector<size_t> shape);


    NodeId im2col(NodeId x, size_t kh, size_t kw, size_t stride, size_t pad);


    NodeId col2im(NodeId cols, size_t C, size_t H, size_t W, size_t kh, size_t kw, size_t stride, size_t pad);
    NodeId maxpool(NodeId x, size_t ph, size_t pw, size_t stride);


    NodeId maxpool_backward(NodeId x, NodeId y, NodeId gy, size_t ph, size_t pw, size_t stride);


    NodeId conv2d(NodeId x, NodeId kernel, size_t stride, size_t pad);
    NodeId softmax(NodeId x);
    NodeId cross_entropy(NodeId pred, NodeId target);
    NodeId batch_norm(NodeId x, NodeId gamma, NodeId beta, double eps);


    NodeId binop(OpKind kind, NodeId a, NodeId b);


    const Node& node(NodeId id) const { return nodes_[id]; }
    Node& node_mut(NodeId id) { return nodes_[id]; }
    size_t size() const { return nodes_.size(); }
    const std::vector<NodeId>& inputs() const { return inputs_; }
    size_t num_inputs() const { return inputs_.size(); }
    const std::vector<size_t>& shape_of(NodeId id) const { return nodes_[id].shape; }


    NodeId push_raw(Node n);

private:
    std::vector<Node> nodes_;
    std::vector<NodeId> inputs_;
};

}
