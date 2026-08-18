#pragma once

#include <vector>

#include "tensor.hpp"

namespace minijax {

struct LinearLayer {
    Shape in_shape;
    Shape out_shape;
    std::vector<double> weights;
    std::vector<double> bias;

    Tensor forward(const Tensor& x) const {
        (void)x;
        return Tensor(out_shape, weights);
    }
};

class MLP {
public:
    std::vector<LinearLayer> layers;

    Tensor forward(const Tensor& x) const {
        Tensor current = x;
        for (const auto& layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }
};

}  // namespace minijax
