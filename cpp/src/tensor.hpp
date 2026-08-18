#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace minijax {

using Shape = std::vector<size_t>;

inline size_t product(const Shape& shape) {
    size_t out = 1;
    for (size_t v : shape) out *= v;
    return out;
}

inline std::vector<size_t> broadcast_shapes(const Shape& a, const Shape& b) {
    size_t n = std::max(a.size(), b.size());
    std::vector<size_t> out(n, 1);
    for (size_t i = 0; i < n; ++i) {
        size_t da = (i + a.size() < n) ? 1 : a[i + a.size() - n];
        size_t db = (i + b.size() < n) ? 1 : b[i + b.size() - n];
        if (da == db) {
            out[i] = da;
        } else if (da == 1) {
            out[i] = db;
        } else if (db == 1) {
            out[i] = da;
        } else {
            throw std::runtime_error("incompatible broadcast shapes");
        }
    }
    return out;
}

inline std::vector<size_t> reshape_index(const Shape& shape, size_t flat_index) {
    std::vector<size_t> idx(shape.size(), 0);
    size_t remaining = flat_index;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        size_t dim = shape[i];
        if (dim == 0) {
            idx[i] = 0;
            continue;
        }
        idx[i] = remaining % dim;
        remaining /= dim;
    }
    return idx;
}

struct Tensor {
    Shape shape;
    std::vector<double> data;

    Tensor() = default;

    Tensor(Shape s, std::vector<double> d) : shape(std::move(s)), data(std::move(d)) {
        if (product(shape) != data.size()) {
            throw std::runtime_error("tensor size mismatch");
        }
    }

    static Tensor scalar(double v) {
        return Tensor({}, {v});
    }

    static Tensor zeros(const Shape& s) {
        return Tensor(s, std::vector<double>(product(s), 0.0));
    }

    static Tensor full(const Shape& s, double v) {
        return Tensor(s, std::vector<double>(product(s), v));
    }

    static Tensor from_matrix(std::vector<std::vector<double>> rows) {
        if (rows.empty()) {
            return Tensor({0}, {});
        }
        size_t cols = rows[0].size();
        for (const auto& row : rows) {
            if (row.size() != cols) {
                throw std::runtime_error("ragged matrix");
            }
        }
        Shape shape = {rows.size(), cols};
        std::vector<double> flat;
        flat.reserve(rows.size() * cols);
        for (const auto& row : rows) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        return Tensor(shape, flat);
    }

    bool is_scalar() const { return shape.empty(); }
    size_t numel() const { return product(shape); }
    double scalar() const {
        if (data.empty()) {
            throw std::runtime_error("scalar tensor has no data");
        }
        return data[0];
    }

    double& at(const std::vector<size_t>& idx) {
        size_t pos = 0;
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            pos += idx[static_cast<size_t>(i)] * stride;
            stride *= shape[static_cast<size_t>(i)];
        }
        return data[pos];
    }

    const double& at(const std::vector<size_t>& idx) const {
        size_t pos = 0;
        size_t stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            pos += idx[static_cast<size_t>(i)] * stride;
            stride *= shape[static_cast<size_t>(i)];
        }
        return data[pos];
    }

    double& at_1d(size_t i) {
        return data[i];
    }

    const double& at_1d(size_t i) const {
        return data[i];
    }

    Tensor reshape(const Shape& new_shape) const {
        if (product(new_shape) != data.size()) {
            throw std::runtime_error("reshape size mismatch");
        }
        return Tensor(new_shape, data);
    }

    Tensor transpose() const {
        if (shape.size() != 2) {
            throw std::runtime_error("transpose expects 2D tensor");
        }
        Shape out = {shape[1], shape[0]};
        std::vector<double> out_data(shape[0] * shape[1]);
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                out_data[j * shape[0] + i] = data[i * shape[1] + j];
            }
        }
        return Tensor(out, out_data);
    }

    Tensor sum() const {
        double total = 0.0;
        for (double v : data) total += v;
        return Tensor({}, {total});
    }

    Tensor sum_axis(size_t axis) const {
        if (shape.empty()) {
            return *this;
        }
        if (axis >= shape.size()) {
            throw std::runtime_error("sum_axis out of range");
        }
        Shape out_shape;
        out_shape.reserve(shape.size() - 1);
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != axis) out_shape.push_back(shape[i]);
        }
        if (out_shape.empty()) {
            return Tensor({}, {sum().data[0]});
        }

        std::vector<double> out(product(out_shape), 0.0);
        size_t outer = 1;
        for (size_t i = 0; i < axis; ++i) outer *= shape[i];
        size_t inner = 1;
        for (size_t i = axis + 1; i < shape.size(); ++i) inner *= shape[i];
        size_t axis_size = shape[axis];

        for (size_t idx = 0; idx < outer; ++idx) {
            for (size_t j = 0; j < inner; ++j) {
                size_t base = idx * axis_size * inner + j;
                double acc = 0.0;
                for (size_t k = 0; k < axis_size; ++k) {
                    acc += data[base + k * inner];
                }
                out[idx * inner + j] = acc;
            }
        }
        return Tensor(out_shape, out);
    }

    Tensor broadcast_to(const Shape& target) const {
        if (shape == target) {
            return *this;
        }
        if (shape.size() == 0) {
            return Tensor(target, std::vector<double>(product(target), data[0]));
        }
        Shape src = shape;
        while (src.size() < target.size()) src.insert(src.begin(), 1);
        std::vector<double> out_data(product(target), 0.0);
        std::vector<size_t> idx(src.size(), 0);
        size_t total = product(target);
        for (size_t flat = 0; flat < total; ++flat) {
            auto cur = reshape_index(target, flat);
            for (size_t i = 0; i < src.size(); ++i) {
                idx[i] = (src[i] == 1) ? 0 : cur[i];
            }
            size_t src_flat = 0;
            size_t stride = 1;
            for (int i = static_cast<int>(src.size()) - 1; i >= 0; --i) {
                src_flat += idx[i] * stride;
                stride *= src[i];
            }
            out_data[flat] = data[src_flat];
        }
        return Tensor(target, out_data);
    }

    Tensor map(std::function<double(double)> f) const {
        std::vector<double> out(data.size());
        std::transform(data.begin(), data.end(), out.begin(), f);
        return Tensor(shape, out);
    }

    Tensor matmul(const Tensor& other) const {
        if (shape.size() != 2 || other.shape.size() != 2) {
            throw std::runtime_error("matmul expects 2D inputs");
        }
        if (shape[1] != other.shape[0]) {
            throw std::runtime_error("matmul inner dimension mismatch");
        }
        size_t m = shape[0];
        size_t k = shape[1];
        size_t n = other.shape[1];
        std::vector<double> out(m * n, 0.0);
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double acc = 0.0;
                for (size_t p = 0; p < k; ++p) {
                    acc += data[i * k + p] * other.data[p * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        return Tensor({m, n}, out);
    }

    Tensor relu() const { return map([](double x) { return x > 0.0 ? x : 0.0; }); }
    Tensor step() const { return map([](double x) { return x > 0.0 ? 1.0 : 0.0; }); }
    Tensor exp() const { return map([](double x) { return std::exp(x); }); }
    Tensor log() const { return map([](double x) { return std::log(x); }); }
    Tensor tanh() const { return map([](double x) { return std::tanh(x); }); }
    Tensor sigmoid() const { return map([](double x) { return 1.0 / (1.0 + std::exp(-x)); }); }

    friend Tensor operator+(const Tensor& a, const Tensor& b) {
        if (a.shape != b.shape) {
            throw std::runtime_error("elementwise add requires same shape");
        }
        std::vector<double> out(a.data.size());
        for (size_t i = 0; i < a.data.size(); ++i) out[i] = a.data[i] + b.data[i];
        return Tensor(a.shape, out);
    }

    friend Tensor operator-(const Tensor& a, const Tensor& b) {
        if (a.shape != b.shape) {
            throw std::runtime_error("elementwise sub requires same shape");
        }
        std::vector<double> out(a.data.size());
        for (size_t i = 0; i < a.data.size(); ++i) out[i] = a.data[i] - b.data[i];
        return Tensor(a.shape, out);
    }

    friend Tensor operator*(const Tensor& a, const Tensor& b) {
        if (a.shape != b.shape) {
            throw std::runtime_error("elementwise mul requires same shape");
        }
        std::vector<double> out(a.data.size());
        for (size_t i = 0; i < a.data.size(); ++i) out[i] = a.data[i] * b.data[i];
        return Tensor(a.shape, out);
    }

    friend Tensor operator/(const Tensor& a, const Tensor& b) {
        if (a.shape != b.shape) {
            throw std::runtime_error("elementwise div requires same shape");
        }
        std::vector<double> out(a.data.size());
        for (size_t i = 0; i < a.data.size(); ++i) out[i] = a.data[i] / b.data[i];
        return Tensor(a.shape, out);
    }

    friend Tensor operator-(const Tensor& t) {
        std::vector<double> out(t.data.size());
        for (size_t i = 0; i < t.data.size(); ++i) out[i] = -t.data[i];
        return Tensor(t.shape, out);
    }

    std::string to_string() const {
        std::ostringstream os;
        os << "shape=[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) os << ", ";
            os << shape[i];
        }
        os << "] data=[";
        for (size_t i = 0; i < data.size(); ++i) {
            if (i) os << ", ";
            os << data[i];
        }
        os << "]";
        return os.str();
    }
};

}  // namespace minijax
