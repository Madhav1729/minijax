#include <gtest/gtest.h>
#include "minijax/vm.hpp"
#include "minijax/interp.hpp"
#include "minijax/autodiff.hpp"
#include "fixtures.hpp"

using namespace minijax;

TEST(Vm, MatchesInterpreterOnLossFixture) {
    auto f = test_fixtures::make_loss_fixture();
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    Tensor expected = eval(f.g, inputs, f.loss);

    Program prog = compile(f.g, f.loss);
    Tensor got = run_vm(prog, inputs);
    EXPECT_TRUE(Tensor::allclose(expected, got, 1e-9, 1e-9));
}

TEST(Vm, MatchesInterpreterAfterCompact) {
    auto f = test_fixtures::make_loss_fixture();
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    Tensor expected = eval(f.g, inputs, f.loss);

    Program prog = compact(compile(f.g, f.loss));
    Tensor got = run_vm(prog, inputs);
    EXPECT_TRUE(Tensor::allclose(expected, got, 1e-9, 1e-9));
}

TEST(Vm, MatchesInterpreterOnGradientGraph) {
    auto f = test_fixtures::make_loss_fixture();
    auto grads = grad(f.g, f.loss, {f.W, f.x, f.y});
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    for (NodeId gid : grads) {
        Tensor expected = eval(f.g, inputs, gid);
        Program prog = compact(compile(f.g, gid));
        Tensor got = run_vm(prog, inputs);
        EXPECT_TRUE(Tensor::allclose(expected, got, 1e-9, 1e-6));
    }
}

TEST(Vm, CompactShrinksRegisterCountOnLongChain) {


    Graph g;
    NodeId cur = g.input({4});
    for (int i = 0; i < 20; ++i) cur = g.relu(cur);

    Program naive = compile(g, cur);
    Program compacted = compact(naive);

    EXPECT_EQ(naive.num_regs, g.size());
    EXPECT_LT(compacted.num_regs, naive.num_regs);
    EXPECT_LE(compacted.num_regs, 3u);

    Tensor input = Tensor::from_vec({4}, {-1, 2, -3, 4});
    Tensor expected = eval(g, {input}, cur);
    Tensor got = run_vm(compacted, {input});
    EXPECT_TRUE(Tensor::allclose(expected, got));
}

TEST(Vm, CompactNeverAliasesLiveRegisters) {


    Graph g;
    NodeId a = g.input({3});
    NodeId b = g.relu(a);
    NodeId c = g.exp(a);
    NodeId d = g.tanh(c);
    NodeId e = g.sigmoid(d);
    NodeId f2 = g.add(b, e);
    NodeId loss = g.sum(f2);

    Tensor input = Tensor::from_vec({3}, {0.3, -0.7, 1.1});
    Tensor expected = eval(g, {input}, loss);

    Program compacted = compact(compile(g, loss));
    Tensor got = run_vm(compacted, {input});
    EXPECT_TRUE(Tensor::allclose(expected, got, 1e-9, 1e-6));
}

TEST(Vm, SerializationRoundTrips) {
    auto f = test_fixtures::make_loss_fixture();
    Program prog = compact(compile(f.g, f.loss));
    std::vector<uint8_t> bytes = serialize(prog);
    Program restored = deserialize(bytes);
    EXPECT_TRUE(prog == restored);


    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    Tensor expected = eval(f.g, inputs, f.loss);
    Tensor got = run_vm(restored, inputs);
    EXPECT_TRUE(Tensor::allclose(expected, got, 1e-9, 1e-9));
}

TEST(Vm, DeserializeTruncatedBufferThrows) {
    auto f = test_fixtures::make_loss_fixture();
    Program prog = compact(compile(f.g, f.loss));
    std::vector<uint8_t> bytes = serialize(prog);
    bytes.resize(bytes.size() / 2);
    EXPECT_THROW(deserialize(bytes), std::runtime_error);
}

TEST(Vm, UninitializedRegisterReadThrows) {
    Program p;
    p.num_regs = 2;
    p.output_reg = 1;
    EXPECT_THROW(run_vm(p, {}), std::runtime_error);
}
