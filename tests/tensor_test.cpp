#include <gtest/gtest.h>
#include "minijax/tensor.hpp"

using minijax::Tensor;

TEST(Tensor, ScalarIsRankZero) {
    Tensor s = Tensor::scalar(3.5);
    EXPECT_EQ(s.rank(), 0u);
    EXPECT_EQ(s.numel(), 1u);
    EXPECT_DOUBLE_EQ(s.item(), 3.5);
}

TEST(Tensor, ElementwiseAddMulMatchHandComputed) {
    Tensor a = Tensor::from_vec({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from_vec({2, 2}, {10, 20, 30, 40});
    Tensor sum = a + b;
    Tensor expected_sum = Tensor::from_vec({2, 2}, {11, 22, 33, 44});
    EXPECT_TRUE(Tensor::allclose(sum, expected_sum));

    Tensor prod = a * b;
    Tensor expected_prod = Tensor::from_vec({2, 2}, {10, 40, 90, 160});
    EXPECT_TRUE(Tensor::allclose(prod, expected_prod));
}

TEST(Tensor, ShapeMismatchThrows) {
    Tensor a = Tensor::from_vec({2, 2}, {1, 2, 3, 4});
    Tensor b = Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6});
    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(Tensor, ReluAndStep) {
    Tensor x = Tensor::from_vec({4}, {-2, -0.0001, 0.5, 3});
    Tensor r = x.relu();
    Tensor expected_relu = Tensor::from_vec({4}, {0, 0, 0.5, 3});
    EXPECT_TRUE(Tensor::allclose(r, expected_relu));

    Tensor s = x.step();
    Tensor expected_step = Tensor::from_vec({4}, {0, 0, 1, 1});
    EXPECT_TRUE(Tensor::allclose(s, expected_step));
}

TEST(Tensor, MatmulHandComputed3x3) {

    Tensor a = Tensor::from_vec({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Tensor id = Tensor::from_vec({3, 3}, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Tensor r = Tensor::matmul(a, id);
    EXPECT_TRUE(Tensor::allclose(r, a));


    Tensor m1 = Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor m2 = Tensor::from_vec({3, 2}, {7, 8, 9, 10, 11, 12});
    Tensor r2 = Tensor::matmul(m1, m2);


    Tensor expected = Tensor::from_vec({2, 2}, {58, 64, 139, 154});
    EXPECT_TRUE(Tensor::allclose(r2, expected));
}

TEST(Tensor, MatmulInnerDimMismatchThrows) {
    Tensor a = Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b = Tensor::from_vec({4, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    EXPECT_THROW(Tensor::matmul(a, b), std::invalid_argument);
}

TEST(Tensor, TransposeRank2) {
    Tensor a = Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor t = a.transpose();
    Tensor expected = Tensor::from_vec({3, 2}, {1, 4, 2, 5, 3, 6});
    EXPECT_TRUE(Tensor::allclose(t, expected));
}

TEST(Tensor, BroadcastScalarToShape) {
    Tensor s = Tensor::scalar(7.0);
    Tensor b = s.broadcast_to({2, 2});
    Tensor expected = Tensor::full({2, 2}, 7.0);
    EXPECT_TRUE(Tensor::allclose(b, expected));
}

TEST(Tensor, BroadcastRowVectorAcrossRows) {

    Tensor row = Tensor::from_vec({3}, {1, 2, 3});
    Tensor b = row.broadcast_to({2, 3});
    Tensor expected = Tensor::from_vec({2, 3}, {1, 2, 3, 1, 2, 3});
    EXPECT_TRUE(Tensor::allclose(b, expected));
}

TEST(Tensor, SumFullReducesToScalar) {
    Tensor a = Tensor::from_vec({2, 2}, {1, 2, 3, 4});
    Tensor s = a.sum();
    EXPECT_EQ(s.rank(), 0u);
    EXPECT_DOUBLE_EQ(s.item(), 10.0);
}

TEST(Tensor, SumAxisReducesCorrectDim) {
    Tensor a = Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor s0 = a.sum_axis(0);
    Tensor expected0 = Tensor::from_vec({3}, {5, 7, 9});
    EXPECT_TRUE(Tensor::allclose(s0, expected0));

    Tensor s1 = a.sum_axis(1);
    Tensor expected1 = Tensor::from_vec({2}, {6, 15});
    EXPECT_TRUE(Tensor::allclose(s1, expected1));
}

TEST(Tensor, ReshapePreservesData) {
    Tensor a = Tensor::from_vec({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor r = a.reshape({3, 2});
    Tensor expected = Tensor::from_vec({3, 2}, {1, 2, 3, 4, 5, 6});
    EXPECT_TRUE(Tensor::allclose(r, expected));
    EXPECT_THROW(a.reshape({4, 2}), std::invalid_argument);
}
