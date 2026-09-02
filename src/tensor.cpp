#include "minijax/tensor.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <limits>

namespace minijax {

size_t Tensor::numel_of(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

Tensor Tensor::zeros(std::vector<size_t> shape) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.data_.assign(numel_of(t.shape_), 0.0);
    return t;
}

Tensor Tensor::full(std::vector<size_t> shape, double value) {
    Tensor t;
    t.shape_ = std::move(shape);
    t.data_.assign(numel_of(t.shape_), value);
    return t;
}

Tensor Tensor::from_vec(std::vector<size_t> shape, std::vector<double> data) {
    if (data.size() != numel_of(shape)) {
        throw std::invalid_argument("Tensor::from_vec: data size does not match shape");
    }
    Tensor t;
    t.shape_ = std::move(shape);
    t.data_ = std::move(data);
    return t;
}

Tensor Tensor::scalar(double value) {
    Tensor t;
    t.shape_ = {};
    t.data_ = {value};
    return t;
}

static size_t flat_index(const std::vector<size_t>& shape, const std::vector<size_t>& idx) {
    if (idx.size() != shape.size()) {
        throw std::invalid_argument("Tensor::at: index rank does not match tensor rank");
    }
    size_t flat = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (idx[i] >= shape[i]) throw std::out_of_range("Tensor::at: index out of range");
        flat = flat * shape[i] + idx[i];
    }
    return flat;
}

double& Tensor::at(const std::vector<size_t>& idx) {
    return data_[flat_index(shape_, idx)];
}

double Tensor::at(const std::vector<size_t>& idx) const {
    return data_[flat_index(shape_, idx)];
}

double Tensor::item() const {
    if (rank() != 0) {
        throw std::invalid_argument("Tensor::item: called on non-scalar (rank " +
                                     std::to_string(rank()) + ")");
    }
    return data_[0];
}

void Tensor::require_same_shape(const Tensor& other, const char* op) const {
    if (shape_ != other.shape_) {
        std::ostringstream os;
        os << "Tensor::" << op << ": shape mismatch";
        throw std::invalid_argument(os.str());
    }
}

Tensor Tensor::mapv(const std::function<double(double)>& f) const {
    Tensor out;
    out.shape_ = shape_;
    out.data_.resize(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) out.data_[i] = f(data_[i]);
    return out;
}

Tensor Tensor::zip_map(const Tensor& a, const Tensor& b,
                        const std::function<double(double, double)>& f) {
    a.require_same_shape(b, "zip_map");
    Tensor out;
    out.shape_ = a.shape_;
    out.data_.resize(a.data_.size());
    for (size_t i = 0; i < a.data_.size(); ++i) out.data_[i] = f(a.data_[i], b.data_[i]);
    return out;
}

Tensor Tensor::operator+(const Tensor& o) const { return zip_map(*this, o, std::plus<>{}); }
Tensor Tensor::operator-(const Tensor& o) const { return zip_map(*this, o, std::minus<>{}); }
Tensor Tensor::operator*(const Tensor& o) const { return zip_map(*this, o, std::multiplies<>{}); }
Tensor Tensor::operator/(const Tensor& o) const { return zip_map(*this, o, std::divides<>{}); }
Tensor Tensor::operator-() const { return mapv([](double x) { return -x; }); }

Tensor Tensor::relu() const { return mapv([](double x) { return x > 0.0 ? x : 0.0; }); }
Tensor Tensor::step() const { return mapv([](double x) { return x > 0.0 ? 1.0 : 0.0; }); }
Tensor Tensor::exp() const { return mapv([](double x) { return std::exp(x); }); }
Tensor Tensor::log() const { return mapv([](double x) { return std::log(x); }); }
Tensor Tensor::tanh() const { return mapv([](double x) { return std::tanh(x); }); }
Tensor Tensor::sigmoid() const { return mapv([](double x) { return 1.0 / (1.0 + std::exp(-x)); }); }
Tensor Tensor::sqrt() const { return mapv([](double x) { return std::sqrt(x); }); }
Tensor Tensor::abs() const { return mapv([](double x) { return std::fabs(x); }); }

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
    if (numel_of(new_shape) != data_.size()) {
        throw std::invalid_argument("Tensor::reshape: element count mismatch");
    }
    Tensor out;
    out.shape_ = std::move(new_shape);
    out.data_ = data_;
    return out;
}

Tensor Tensor::transpose() const {
    if (rank() != 2) {
        throw std::invalid_argument("Tensor::transpose: only rank-2 tensors are supported");
    }
    size_t rows = shape_[0], cols = shape_[1];
    Tensor out = Tensor::zeros({cols, rows});
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out.data_[j * rows + i] = data_[i * cols + j];
    return out;
}

Tensor Tensor::broadcast_to(std::vector<size_t> target_shape) const {


    if (rank() == 0) {
        return Tensor::full(target_shape, data_[0]);
    }
    if (target_shape.size() < shape_.size()) {
        throw std::invalid_argument("Tensor::broadcast_to: target rank smaller than source rank");
    }
    size_t rank_diff = target_shape.size() - shape_.size();

    std::vector<size_t> padded(target_shape.size(), 1);
    for (size_t i = 0; i < shape_.size(); ++i) padded[rank_diff + i] = shape_[i];
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (padded[i] != target_shape[i] && padded[i] != 1) {
            throw std::invalid_argument("Tensor::broadcast_to: incompatible shapes");
        }
    }
    Tensor out = Tensor::zeros(target_shape);
    size_t n = out.data_.size();
    std::vector<size_t> strides(target_shape.size());
    size_t acc = 1;
    for (size_t i = target_shape.size(); i-- > 0;) { strides[i] = acc; acc *= target_shape[i]; }
    std::vector<size_t> src_strides(padded.size());
    acc = 1;
    for (size_t i = padded.size(); i-- > 0;) { src_strides[i] = (padded[i] == 1 ? 0 : acc); acc *= padded[i]; }
    for (size_t flat = 0; flat < n; ++flat) {
        size_t rem = flat, src_flat = 0;
        for (size_t i = 0; i < target_shape.size(); ++i) {
            size_t coord = rem / strides[i];
            rem %= strides[i];
            src_flat += coord * src_strides[i];
        }
        out.data_[flat] = data_[src_flat];
    }
    return out;
}

Tensor Tensor::sum() const {
    double s = std::accumulate(data_.begin(), data_.end(), 0.0);
    return Tensor::scalar(s);
}

Tensor Tensor::sum_axis(size_t axis) const {
    if (axis >= shape_.size()) throw std::invalid_argument("Tensor::sum_axis: axis out of range");
    std::vector<size_t> out_shape;
    for (size_t i = 0; i < shape_.size(); ++i) if (i != axis) out_shape.push_back(shape_[i]);
    Tensor out = Tensor::zeros(out_shape);

    std::vector<size_t> strides(shape_.size());
    size_t acc = 1;
    for (size_t i = shape_.size(); i-- > 0;) { strides[i] = acc; acc *= shape_[i]; }

    for (size_t flat = 0; flat < data_.size(); ++flat) {
        size_t rem = flat;
        std::vector<size_t> coord(shape_.size());
        for (size_t i = 0; i < shape_.size(); ++i) { coord[i] = rem / strides[i]; rem %= strides[i]; }
        size_t out_flat = 0, out_acc_stride = 1;

        std::vector<size_t> out_coord;
        for (size_t i = 0; i < shape_.size(); ++i) if (i != axis) out_coord.push_back(coord[i]);
        std::vector<size_t> out_strides(out_shape.size());
        size_t oacc = 1;
        for (size_t i = out_shape.size(); i-- > 0;) { out_strides[i] = oacc; oacc *= out_shape[i]; }
        for (size_t i = 0; i < out_coord.size(); ++i) out_flat += out_coord[i] * out_strides[i];
        (void)out_acc_stride;
        out.data_[out_flat] += data_[flat];
    }
    return out;
}

Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
    if (a.rank() != 2 || b.rank() != 2) {
        throw std::invalid_argument("Tensor::matmul: both operands must be rank-2");
    }
    size_t m = a.shape_[0], k = a.shape_[1], k2 = b.shape_[0], n = b.shape_[1];
    if (k != k2) {
        throw std::invalid_argument("Tensor::matmul: inner dimensions do not match (" +
                                     std::to_string(k) + " vs " + std::to_string(k2) + ")");
    }
    Tensor out = Tensor::zeros({m, n});


    using RowMajMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMajMat> ea(a.data_.data(), m, k);
    Eigen::Map<const RowMajMat> eb(b.data_.data(), k, n);
    Eigen::Map<RowMajMat> eout(out.data_.data(), m, n);
    eout.noalias() = ea * eb;
    return out;
}

bool Tensor::allclose(const Tensor& a, const Tensor& b, double atol, double rtol) {
    if (a.shape_ != b.shape_) return false;
    for (size_t i = 0; i < a.data_.size(); ++i) {
        double x = a.data_[i], y = b.data_[i];
        if (std::fabs(x - y) > atol + rtol * std::fabs(y)) return false;
    }
    return true;
}


namespace {
struct ConvGeom {
    size_t C, H, W, OH, OW;
};
ConvGeom conv_geom(const std::vector<size_t>& shape, size_t kh, size_t kw, size_t stride, size_t pad) {
    if (shape.size() != 3) throw std::invalid_argument("im2col/maxpool: expected rank-3 [C,H,W] tensor");
    size_t C = shape[0], H = shape[1], W = shape[2];
    if (H + 2 * pad < kh || W + 2 * pad < kw) throw std::invalid_argument("im2col: kernel larger than padded input");
    size_t OH = (H + 2 * pad - kh) / stride + 1;
    size_t OW = (W + 2 * pad - kw) / stride + 1;
    return {C, H, W, OH, OW};
}
}

Tensor Tensor::im2col(size_t kh, size_t kw, size_t stride, size_t pad) const {
    ConvGeom g = conv_geom(shape_, kh, kw, stride, pad);
    Tensor out = Tensor::zeros({g.C * kh * kw, g.OH * g.OW});
    for (size_t oh = 0; oh < g.OH; ++oh) {
        for (size_t ow = 0; ow < g.OW; ++ow) {
            size_t col = oh * g.OW + ow;
            for (size_t c = 0; c < g.C; ++c) {
                for (size_t i = 0; i < kh; ++i) {
                    for (size_t j = 0; j < kw; ++j) {
                        long ih = static_cast<long>(oh * stride + i) - static_cast<long>(pad);
                        long iw = static_cast<long>(ow * stride + j) - static_cast<long>(pad);
                        size_t row = c * kh * kw + i * kw + j;
                        double v = 0.0;
                        if (ih >= 0 && iw >= 0 && static_cast<size_t>(ih) < g.H && static_cast<size_t>(iw) < g.W) {
                            v = data_[(c * g.H + static_cast<size_t>(ih)) * g.W + static_cast<size_t>(iw)];
                        }
                        out.data_[row * (g.OH * g.OW) + col] = v;
                    }
                }
            }
        }
    }
    return out;
}

Tensor Tensor::col2im(const Tensor& cols, size_t C, size_t H, size_t W,
                       size_t kh, size_t kw, size_t stride, size_t pad) {
    ConvGeom g = conv_geom({C, H, W}, kh, kw, stride, pad);
    if (cols.shape_ != std::vector<size_t>{C * kh * kw, g.OH * g.OW}) {
        throw std::invalid_argument("col2im: cols shape does not match expected im2col output shape");
    }
    Tensor out = Tensor::zeros({C, H, W});
    for (size_t oh = 0; oh < g.OH; ++oh) {
        for (size_t ow = 0; ow < g.OW; ++ow) {
            size_t col = oh * g.OW + ow;
            for (size_t c = 0; c < C; ++c) {
                for (size_t i = 0; i < kh; ++i) {
                    for (size_t j = 0; j < kw; ++j) {
                        long ih = static_cast<long>(oh * stride + i) - static_cast<long>(pad);
                        long iw = static_cast<long>(ow * stride + j) - static_cast<long>(pad);
                        if (ih >= 0 && iw >= 0 && static_cast<size_t>(ih) < H && static_cast<size_t>(iw) < W) {
                            size_t row = c * kh * kw + i * kw + j;
                            out.data_[(c * H + static_cast<size_t>(ih)) * W + static_cast<size_t>(iw)] +=
                                cols.data_[row * (g.OH * g.OW) + col];
                        }
                    }
                }
            }
        }
    }
    return out;
}

Tensor Tensor::maxpool(size_t ph, size_t pw, size_t stride) const {
    ConvGeom g = conv_geom(shape_, ph, pw, stride, 0);
    Tensor out = Tensor::zeros({g.C, g.OH, g.OW});
    for (size_t c = 0; c < g.C; ++c) {
        for (size_t oh = 0; oh < g.OH; ++oh) {
            for (size_t ow = 0; ow < g.OW; ++ow) {
                double best = -std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < ph; ++i) {
                    for (size_t j = 0; j < pw; ++j) {
                        size_t ih = oh * stride + i, iw = ow * stride + j;
                        double v = data_[(c * g.H + ih) * g.W + iw];
                        if (v > best) best = v;
                    }
                }
                out.data_[(c * g.OH + oh) * g.OW + ow] = best;
            }
        }
    }
    return out;
}

Tensor Tensor::maxpool_backward(const Tensor& x, const Tensor& y, const Tensor& grad_y,
                                 size_t ph, size_t pw, size_t stride) {
    ConvGeom g = conv_geom(x.shape_, ph, pw, stride, 0);
    if (y.shape_ != std::vector<size_t>{g.C, g.OH, g.OW} || grad_y.shape_ != y.shape_) {
        throw std::invalid_argument("maxpool_backward: shape mismatch between x, y, grad_y");
    }
    Tensor grad_x = Tensor::zeros(x.shape_);
    for (size_t c = 0; c < g.C; ++c) {
        for (size_t oh = 0; oh < g.OH; ++oh) {
            for (size_t ow = 0; ow < g.OW; ++ow) {
                double target = y.data_[(c * g.OH + oh) * g.OW + ow];
                double gy = grad_y.data_[(c * g.OH + oh) * g.OW + ow];
                bool routed = false;
                for (size_t i = 0; i < ph && !routed; ++i) {
                    for (size_t j = 0; j < pw && !routed; ++j) {
                        size_t ih = oh * stride + i, iw = ow * stride + j;
                        size_t idx = (c * g.H + ih) * g.W + iw;
                        if (x.data_[idx] == target) {
                            grad_x.data_[idx] += gy;
                            routed = true;
                        }
                    }
                }
            }
        }
    }
    return grad_x;
}

}
