#include <gtest/gtest.h>
#include "minijax/nn.hpp"
#include "minijax/autodiff.hpp"
#include "minijax/interp.hpp"

using namespace minijax;

TEST(Nn, MlpForwardProducesCorrectShape) {
    Graph g;
    MLP mlp(g, {2, 8, 3}, 1);
    NodeId x = g.input({2, 1});
    NodeId out = mlp.forward(g, x);
    EXPECT_EQ(g.shape_of(out), (std::vector<size_t>{3}));
}

TEST(Nn, ParamsCountMatchesLayerStructure) {
    Graph g;
    MLP mlp(g, {2, 8, 8, 2}, 1);

    EXPECT_EQ(mlp.params.size(), 6u);
}

TEST(Nn, XavierInitIsWithinExpectedRange) {
    Graph g;
    MLP mlp(g, {4, 4}, 7);
    double limit = std::sqrt(6.0 / (4.0 + 4.0));
    const Tensor& w = mlp.params[0].value;
    for (double v : w.data()) {
        EXPECT_GE(v, -limit);
        EXPECT_LE(v, limit);
    }
}

TEST(Nn, SgdStepMovesInNegativeGradientDirection) {
    std::vector<Param> params = {{0, Tensor::from_vec({2}, {1.0, -1.0}), "p"}};
    std::vector<Tensor> grads = {Tensor::from_vec({2}, {0.5, 0.5})};
    SGD opt{0.1};
    opt.step(params, grads);

    EXPECT_TRUE(Tensor::allclose(params[0].value, Tensor::from_vec({2}, {0.95, -1.05})));
}

TEST(Nn, AdamStepDecreasesParamInGradientDirection) {
    std::vector<Param> params = {{0, Tensor::from_vec({1}, {1.0}), "p"}};
    Adam opt;
    opt.lr = 0.1;
    for (int i = 0; i < 5; ++i) {
        std::vector<Tensor> grads = {Tensor::from_vec({1}, {1.0})};
        opt.step(params, grads);
    }

    EXPECT_LT(params[0].value.data()[0], 1.0);
}

TEST(Nn, TwoMoonsDatasetHasBalancedLabels) {
    Dataset ds = generate_two_moons(50, 0.1, 42);
    EXPECT_EQ(ds.xs.size(), 100u);
    EXPECT_EQ(ds.ys.size(), 100u);
    double class0 = 0, class1 = 0;
    for (auto& y : ds.ys) {
        if (y.data()[0] > y.data()[1]) class0++; else class1++;
    }
    EXPECT_EQ(class0, 50);
    EXPECT_EQ(class1, 50);
}

TEST(Nn, TwoMoonsGenerationIsDeterministicGivenSeed) {
    Dataset a = generate_two_moons(20, 0.1, 123);
    Dataset b = generate_two_moons(20, 0.1, 123);
    for (size_t i = 0; i < a.xs.size(); ++i) {
        EXPECT_TRUE(Tensor::allclose(a.xs[i], b.xs[i], 0.0, 0.0));
    }
}


TEST(Nn, AdamLossDecreasesMonotonicallyOnFixedSample) {
    Graph g;
    MLP mlp(g, {2, 8, 2}, 3);
    NodeId x_node = g.input({2, 1});
    NodeId y_node = g.input({2, 1});
    NodeId logits = mlp.forward(g, x_node);
    NodeId probs = g.softmax(logits);
    NodeId y_flat = g.reshape(y_node, {2});
    NodeId loss = g.cross_entropy(probs, y_flat);

    std::vector<NodeId> param_nodes;
    for (auto& p : mlp.params) param_nodes.push_back(p.node);
    auto grads = grad(g, loss, param_nodes);

    Tensor x_val = Tensor::from_vec({2, 1}, {0.5, -0.3});
    Tensor y_val = Tensor::from_vec({2, 1}, {1.0, 0.0});

    Adam opt;
    opt.lr = 0.05;

    double prev_loss = std::numeric_limits<double>::infinity();
    int strictly_decreasing_count = 0;
    const int N = 20;
    for (int step = 0; step < N; ++step) {
        auto values = eval_with_params(g, mlp.params, x_node, x_val, y_node, y_val);
        double cur_loss = values[loss].item();
        if (cur_loss < prev_loss) ++strictly_decreasing_count;
        prev_loss = cur_loss;

        std::vector<Tensor> grad_vals;
        for (NodeId gid : grads) grad_vals.push_back(values[gid]);
        opt.step(mlp.params, grad_vals);
    }


    EXPECT_GE(strictly_decreasing_count, N - 2) << "loss should decrease almost every step on a fixed sample";
}


TEST(Nn, MlpReachesHighAccuracyOnTwoMoons) {
    Graph g;
    MLP mlp(g, {2, 16, 16, 2}, 1);
    NodeId x_node = g.input({2, 1});
    NodeId y_node = g.input({2, 1});
    NodeId logits = mlp.forward(g, x_node);
    NodeId probs = g.softmax(logits);
    NodeId y_flat = g.reshape(y_node, {2});
    NodeId loss = g.cross_entropy(probs, y_flat);

    std::vector<NodeId> param_nodes;
    for (auto& p : mlp.params) param_nodes.push_back(p.node);
    auto grads = grad(g, loss, param_nodes);

    Dataset train = generate_two_moons(80, 0.08, 42);

    Adam opt;
    opt.lr = 0.005;

    for (int epoch = 0; epoch < 80; ++epoch) {
        train_epoch(g, mlp.params, x_node, y_node, loss, grads, train, opt);
    }

    double train_acc = compute_accuracy(g, mlp.params, x_node, y_node, logits, train);
    EXPECT_GE(train_acc, 0.95) << "train accuracy was " << (train_acc * 100) << "%";
}
