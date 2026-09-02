#include "minijax/ir.hpp"
#include <algorithm>
#include <sstream>

namespace minijax {

const char* op_kind_name(OpKind k) {
    switch (k) {
        case OpKind::Input: return "Input";
        case OpKind::Const: return "Const";
        case OpKind::Add: return "Add";
        case OpKind::Sub: return "Sub";
        case OpKind::Mul: return "Mul";
        case OpKind::Div: return "Div";
        case OpKind::Neg: return "Neg";
        case OpKind::MatMul: return "MatMul";
        case OpKind::Relu: return "Relu";
        case OpKind::Step: return "Step";
        case OpKind::Tanh: return "Tanh";
        case OpKind::Sigmoid: return "Sigmoid";
        case OpKind::Exp: return "Exp";
        case OpKind::Log: return "Log";
        case OpKind::Sqrt: return "Sqrt";
        case OpKind::Abs: return "Abs";
        case OpKind::Sum: return "Sum";
        case OpKind::SumAxis: return "SumAxis";
        case OpKind::Transpose: return "Transpose";
        case OpKind::Broadcast: return "Broadcast";
        case OpKind::Reshape: return "Reshape";
        case OpKind::Softmax: return "Softmax";
        case OpKind::CrossEntropy: return "CrossEntropy";
        case OpKind::Conv2d: return "Conv2d";
        case OpKind::Im2Col: return "Im2Col";
        case OpKind::Col2Im: return "Col2Im";
        case OpKind::MaxPool: return "MaxPool";
        case OpKind::MaxPoolBackward: return "MaxPoolBackward";
        case OpKind::BatchNorm: return "BatchNorm";
    }
    return "?";
}

NodeId Graph::push_raw(Node n) {

    NodeId new_id = nodes_.size();
    for (NodeId in : n.inputs) {
        if (in >= new_id) {
            throw std::invalid_argument("Graph::push_raw: node input references a non-earlier index "
                                         "(violates topological-arena invariant)");
        }
    }
    nodes_.push_back(std::move(n));
    return new_id;
}

NodeId Graph::input(std::vector<size_t> shape) {
    Node n;
    n.op = OpKind::Input;
    n.shape = std::move(shape);
    n.input_slot = inputs_.size();
    NodeId id = push_raw(std::move(n));
    inputs_.push_back(id);
    return id;
}

NodeId Graph::constant(double value, std::vector<size_t> shape) {
    Node n;
    n.op = OpKind::Const;
    n.const_value = value;
    n.shape = std::move(shape);
    return push_raw(std::move(n));
}


static std::optional<std::vector<size_t>> broadcast_shape(const std::vector<size_t>& a,
                                                            const std::vector<size_t>& b) {
    size_t rank = std::max(a.size(), b.size());
    std::vector<size_t> out(rank);
    for (size_t i = 0; i < rank; ++i) {
        size_t ad = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        size_t bd = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        if (ad == bd) out[i] = ad;
        else if (ad == 1) out[i] = bd;
        else if (bd == 1) out[i] = ad;
        else return std::nullopt;
    }
    return out;
}

NodeId Graph::broadcast_to(NodeId a, std::vector<size_t> shape) {
    Node n;
    n.op = OpKind::Broadcast;
    n.inputs = {a};
    n.target_shape = shape;
    n.shape = shape;
    return push_raw(std::move(n));
}

NodeId Graph::binop(OpKind kind, NodeId a, NodeId b) {


    std::vector<size_t> sa = nodes_[a].shape;
    std::vector<size_t> sb = nodes_[b].shape;
    NodeId lhs = a, rhs = b;
    if (sa != sb) {


        auto target = broadcast_shape(sa, sb);
        if (!target) {
            std::ostringstream os;
            os << "Graph::binop(" << op_kind_name(kind) << "): incompatible shapes for broadcast";
            throw std::invalid_argument(os.str());
        }
        if (sa != *target) lhs = broadcast_to(a, *target);
        if (sb != *target) rhs = broadcast_to(b, *target);
    }
    Node n;
    n.op = kind;
    n.inputs = {lhs, rhs};
    n.shape = nodes_[lhs].shape;
    return push_raw(std::move(n));
}

NodeId Graph::add(NodeId a, NodeId b) { return binop(OpKind::Add, a, b); }
NodeId Graph::sub(NodeId a, NodeId b) { return binop(OpKind::Sub, a, b); }
NodeId Graph::mul(NodeId a, NodeId b) { return binop(OpKind::Mul, a, b); }
NodeId Graph::div(NodeId a, NodeId b) { return binop(OpKind::Div, a, b); }

NodeId Graph::neg(NodeId a) {
    Node n; n.op = OpKind::Neg; n.inputs = {a}; n.shape = nodes_[a].shape;
    return push_raw(std::move(n));
}

NodeId Graph::matmul(NodeId a, NodeId b) {
    const auto& sa = nodes_[a].shape;
    const auto& sb = nodes_[b].shape;
    if (sa.size() != 2 || sb.size() != 2) {
        throw std::invalid_argument("Graph::matmul: both operands must be rank-2");
    }
    if (sa[1] != sb[0]) {
        std::ostringstream os;
        os << "Graph::matmul: inner dim mismatch (" << sa[1] << " vs " << sb[0] << ")";
        throw std::invalid_argument(os.str());
    }
    Node n;
    n.op = OpKind::MatMul;
    n.inputs = {a, b};
    n.shape = {sa[0], sb[1]};
    return push_raw(std::move(n));
}

#define UNARY_SAME_SHAPE_IMPL(Name, Kind) \
    NodeId Graph::Name(NodeId a) { \
        Node n; n.op = OpKind::Kind; n.inputs = {a}; n.shape = nodes_[a].shape; \
        return push_raw(std::move(n)); \
    }

UNARY_SAME_SHAPE_IMPL(relu, Relu)
UNARY_SAME_SHAPE_IMPL(step, Step)
UNARY_SAME_SHAPE_IMPL(tanh, Tanh)
UNARY_SAME_SHAPE_IMPL(sigmoid, Sigmoid)
UNARY_SAME_SHAPE_IMPL(exp, Exp)
UNARY_SAME_SHAPE_IMPL(log, Log)
UNARY_SAME_SHAPE_IMPL(sqrt, Sqrt)
UNARY_SAME_SHAPE_IMPL(abs, Abs)

#undef UNARY_SAME_SHAPE_IMPL

NodeId Graph::sum(NodeId a) {
    Node n; n.op = OpKind::Sum; n.inputs = {a}; n.shape = {};
    return push_raw(std::move(n));
}

NodeId Graph::sum_axis(NodeId a, size_t axis) {
    const auto& sa = nodes_[a].shape;
    if (axis >= sa.size()) throw std::invalid_argument("Graph::sum_axis: axis out of range");
    Node n;
    n.op = OpKind::SumAxis;
    n.inputs = {a};
    n.axis = axis;
    for (size_t i = 0; i < sa.size(); ++i) if (i != axis) n.shape.push_back(sa[i]);
    return push_raw(std::move(n));
}

NodeId Graph::transpose(NodeId a) {
    const auto& sa = nodes_[a].shape;
    if (sa.size() != 2) throw std::invalid_argument("Graph::transpose: only rank-2 supported");
    Node n;
    n.op = OpKind::Transpose;
    n.inputs = {a};
    n.shape = {sa[1], sa[0]};
    return push_raw(std::move(n));
}

NodeId Graph::reshape(NodeId a, std::vector<size_t> shape) {
    const auto& sa = nodes_[a].shape;
    size_t old_n = 1; for (auto d : sa) old_n *= d;
    size_t new_n = 1; for (auto d : shape) new_n *= d;
    if (old_n != new_n) throw std::invalid_argument("Graph::reshape: element count mismatch");
    Node n;
    n.op = OpKind::Reshape;
    n.inputs = {a};
    n.target_shape = shape;
    n.shape = shape;
    return push_raw(std::move(n));
}


namespace {
struct ConvShapeOut { size_t C, OH, OW; };
ConvShapeOut conv_shape_out(const std::vector<size_t>& x_shape, size_t kh, size_t kw, size_t stride, size_t pad) {
    if (x_shape.size() != 3) throw std::invalid_argument("im2col/maxpool: expected rank-3 [C,H,W] input");
    size_t C = x_shape[0], H = x_shape[1], W = x_shape[2];
    if (H + 2 * pad < kh || W + 2 * pad < kw)
        throw std::invalid_argument("im2col/maxpool: kernel larger than padded input");
    size_t OH = (H + 2 * pad - kh) / stride + 1;
    size_t OW = (W + 2 * pad - kw) / stride + 1;
    return {C, OH, OW};
}
}

NodeId Graph::im2col(NodeId x, size_t kh, size_t kw, size_t stride, size_t pad) {
    ConvShapeOut o = conv_shape_out(nodes_[x].shape, kh, kw, stride, pad);
    Node n;
    n.op = OpKind::Im2Col;
    n.inputs = {x};
    n.shape = {o.C * kh * kw, o.OH * o.OW};
    n.kernel_h = kh; n.kernel_w = kw; n.stride = stride; n.pad = pad;
    return push_raw(std::move(n));
}

NodeId Graph::col2im(NodeId cols, size_t C, size_t H, size_t W, size_t kh, size_t kw, size_t stride, size_t pad) {
    ConvShapeOut o = conv_shape_out({C, H, W}, kh, kw, stride, pad);
    const auto& sc = nodes_[cols].shape;
    if (sc != std::vector<size_t>{C * kh * kw, o.OH * o.OW}) {
        throw std::invalid_argument("Graph::col2im: cols shape does not match expected im2col output shape");
    }
    Node n;
    n.op = OpKind::Col2Im;
    n.inputs = {cols};
    n.shape = {C, H, W};
    n.kernel_h = kh; n.kernel_w = kw; n.stride = stride; n.pad = pad;
    return push_raw(std::move(n));
}

NodeId Graph::maxpool(NodeId x, size_t ph, size_t pw, size_t stride) {
    ConvShapeOut o = conv_shape_out(nodes_[x].shape, ph, pw, stride, 0);
    Node n;
    n.op = OpKind::MaxPool;
    n.inputs = {x};
    n.shape = {o.C, o.OH, o.OW};
    n.kernel_h = ph; n.kernel_w = pw; n.stride = stride; n.pad = 0;
    return push_raw(std::move(n));
}

NodeId Graph::maxpool_backward(NodeId x, NodeId y, NodeId gy, size_t ph, size_t pw, size_t stride) {
    if (nodes_[y].shape != nodes_[gy].shape) {
        throw std::invalid_argument("Graph::maxpool_backward: y and gy shape mismatch");
    }
    Node n;
    n.op = OpKind::MaxPoolBackward;
    n.inputs = {x, y, gy};
    n.shape = nodes_[x].shape;
    n.kernel_h = ph; n.kernel_w = pw; n.stride = stride;
    return push_raw(std::move(n));
}


NodeId Graph::conv2d(NodeId x, NodeId kernel, size_t stride, size_t pad) {
    const auto& xs = nodes_[x].shape;
    const auto& ks = nodes_[kernel].shape;
    if (xs.size() != 3) throw std::invalid_argument("Graph::conv2d: x must be rank-3 [C,H,W]");
    if (ks.size() != 4) throw std::invalid_argument("Graph::conv2d: kernel must be rank-4 [OutC,C,KH,KW]");
    size_t C = xs[0];
    size_t OutC = ks[0], KC = ks[1], KH = ks[2], KW = ks[3];
    if (KC != C) throw std::invalid_argument("Graph::conv2d: kernel channel count must match x's channels");

    ConvShapeOut o = conv_shape_out(xs, KH, KW, stride, pad);
    NodeId cols = im2col(x, KH, KW, stride, pad);
    NodeId kernel2d = reshape(kernel, {OutC, C * KH * KW});
    NodeId prod = matmul(kernel2d, cols);
    return reshape(prod, {OutC, o.OH, o.OW});
}

NodeId Graph::softmax(NodeId x) {


    NodeId e = exp(x);
    NodeId s = sum(e);
    NodeId s_b = broadcast_to(s, nodes_[x].shape);
    return div(e, s_b);
}

NodeId Graph::cross_entropy(NodeId pred, NodeId target) {
    NodeId lp = log(pred);
    NodeId prod = mul(target, lp);
    NodeId s = sum(prod);
    return neg(s);
}

NodeId Graph::batch_norm(NodeId x, NodeId gamma, NodeId beta, double eps) {


    std::vector<size_t> xs = nodes_[x].shape;
    size_t n_elems = 1; for (auto d : xs) n_elems *= d;
    NodeId inv_n = constant(1.0 / static_cast<double>(n_elems), {});

    NodeId sum_x = sum(x);
    NodeId mean = mul(sum_x, inv_n);
    NodeId mean_b = broadcast_to(mean, xs);
    NodeId centered = sub(x, mean_b);
    NodeId sq = mul(centered, centered);
    NodeId sum_sq = sum(sq);
    NodeId var = mul(sum_sq, inv_n);
    NodeId var_eps = add(var, constant(eps, {}));
    NodeId std_dev = sqrt(var_eps);
    NodeId std_b = broadcast_to(std_dev, xs);
    NodeId normalized = div(centered, std_b);
    NodeId scaled = mul(gamma, normalized);
    return add(scaled, beta);
}

}
