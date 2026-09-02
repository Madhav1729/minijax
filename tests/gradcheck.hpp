#pragma once


#include <gtest/gtest.h>
#include "minijax/ir.hpp"
#include "minijax/interp.hpp"
#include "minijax/autodiff.hpp"

namespace minijax::test_fixtures {


inline void check_gradient(Graph& g, NodeId output, NodeId wrt,
                            std::vector<Tensor> inputs, double eps = 1e-4, double tol = 1e-4) {
    std::vector<NodeId> grads = grad(g, output, {wrt});
    Tensor analytic = eval(g, inputs, grads[0]);

    size_t slot = g.node(wrt).input_slot;
    Tensor& perturb_target = inputs[slot];
    Tensor numeric = Tensor::zeros(perturb_target.shape());

    for (size_t i = 0; i < perturb_target.numel(); ++i) {
        double orig = perturb_target.data()[i];

        perturb_target.data()[i] = orig + eps;
        double f_plus = eval(g, inputs, output).item();

        perturb_target.data()[i] = orig - eps;
        double f_minus = eval(g, inputs, output).item();

        perturb_target.data()[i] = orig;
        numeric.data()[i] = (f_plus - f_minus) / (2.0 * eps);
    }

    ASSERT_EQ(analytic.shape(), numeric.shape());
    for (size_t i = 0; i < analytic.numel(); ++i) {
        EXPECT_NEAR(analytic.data()[i], numeric.data()[i], tol)
            << "mismatch at element " << i;
    }
}

}
