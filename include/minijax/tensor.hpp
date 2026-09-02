#pragma once


#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <functional>

namespace minijax {

class Tensor {
public:
    Tensor() = default;


    static Tensor zeros(std::vector<size_t> shape);
    static Tensor full(std::vector<size_t> shape, double value);
    static Tensor from_vec(std::vector<size_t> shape, std::vector<double> data);
    static Tensor scalar(double value);


    const std::vector<size_t>& shape() const { return shape_; }
    size_t rank() const { return shape_.size(); }
    size_t numel() const { return data_.size(); }
    const std::vector<double>& data() const { return data_; }
    std::vector<double>& data() { return data_; }

    double& at(const std::vector<size_t>& idx);
    double at(const std::vector<size_t>& idx) const;


    double item() const;

    bool same_shape(const Tensor& other) const { return shape_ == other.shape_; }


    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator-() const;

    Tensor relu() const;
    Tensor step() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor tanh() const;
    Tensor sigmoid() const;
    Tensor sqrt() const;
    Tensor abs() const;


    Tensor mapv(const std::function<double(double)>& f) const;

    static Tensor zip_map(const Tensor& a, const Tensor& b,
                           const std::function<double(double, double)>& f);


    Tensor reshape(std::vector<size_t> new_shape) const;
    Tensor transpose() const;
    Tensor broadcast_to(std::vector<size_t> target_shape) const;


    Tensor sum() const;
    Tensor sum_axis(size_t axis) const;


    static Tensor matmul(const Tensor& a, const Tensor& b);


    Tensor im2col(size_t kh, size_t kw, size_t stride, size_t pad) const;


    static Tensor col2im(const Tensor& cols, size_t C, size_t H, size_t W,
                          size_t kh, size_t kw, size_t stride, size_t pad);

    Tensor maxpool(size_t ph, size_t pw, size_t stride) const;


    static Tensor maxpool_backward(const Tensor& x, const Tensor& y, const Tensor& grad_y,
                                    size_t ph, size_t pw, size_t stride);


    static bool allclose(const Tensor& a, const Tensor& b, double atol = 1e-9, double rtol = 1e-6);

private:
    std::vector<double> data_;
    std::vector<size_t> shape_;

    static size_t numel_of(const std::vector<size_t>& shape);
    void require_same_shape(const Tensor& other, const char* op) const;
};

}
