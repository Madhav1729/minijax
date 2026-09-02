#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "minijax/jit.hpp"
#include "minijax/vm.hpp"
#include "minijax/interp.hpp"
#include "minijax/autodiff.hpp"
#include "fixtures.hpp"

using namespace minijax;


namespace {
std::vector<Tensor> core_inputs() {
    return {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({4}, {1.0, 2.0, 0.5, 4.0}),
    };
}
}

TEST(Jit, MatchesInterpreterOnLossFixture) {
    auto f = test_fixtures::make_loss_fixture();
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    Tensor expected = eval(f.g, inputs, f.loss);
    Tensor got = jit::run_jit(f.g, inputs, f.loss);
    EXPECT_TRUE(Tensor::allclose(expected, got, 1e-9, 1e-9));
}

TEST(Jit, MatchesInterpreterAndVmOnEveryCoreOp) {
    Graph g;
    NodeId m22 = g.input({2, 2});
    NodeId v4 = g.input({4});


    NodeId e_add = g.add(m22, g.constant(1.25, {2, 2}));
    NodeId e_sub = g.sub(e_add, g.constant(0.5, {2, 2}));
    NodeId e_mul = g.mul(e_sub, m22);
    NodeId e_div = g.div(e_mul, g.constant(2.0, {2, 2}));


    NodeId bcast = g.sum_axis(v4, 0);


    NodeId colsum = g.sum_axis(g.transpose(m22), 0);
    NodeId badd = g.add(colsum, g.broadcast_to(g.constant(-0.25), {2}));


    NodeId pos = g.abs(v4);
    NodeId u_exp = g.exp(pos);
    NodeId u_log = g.log(u_exp);
    NodeId u_sqrt = g.sqrt(pos);
    NodeId u_tanh = g.tanh(pos);
    NodeId u_sig = g.sigmoid(u_tanh);
    NodeId u_neg = g.neg(u_sig);
    NodeId u_relu = g.relu(u_neg);
    NodeId u_step = g.step(v4);


    NodeId mm = g.matmul(m22, g.reshape(colsum, {2, 1}));
    NodeId mmt = g.transpose(mm);


    NodeId s_full = g.sum(mm);
    NodeId s_axis = g.sum_axis(mmt, 1);
    NodeId s_rs = g.reshape(s_axis, {});

    std::vector<NodeId> outputs = {e_add,   e_sub, e_mul, e_div,  badd,
                                    colsum, u_exp, u_log, u_sqrt, u_tanh,
                                    u_sig,  u_neg, u_relu, u_step, mm,
                                    mmt,    s_full, s_axis, s_rs};

    auto inputs = core_inputs();

    jit::JitProgram prog(g);
    prog.execute(inputs);

    for (NodeId out : outputs) {
        Tensor want_i = eval(g, inputs, out);
        Tensor got = prog.value(out);
        EXPECT_TRUE(Tensor::allclose(want_i, got, 1e-9, 1e-9))
            << "jit mismatch vs interp at node " << out;

        Tensor want_vm = run_vm(compile(g, out), inputs);
        EXPECT_TRUE(Tensor::allclose(want_vm, got, 1e-9, 1e-9))
            << "jit mismatch vs vm at node " << out;


        prog.execute(inputs);
        EXPECT_TRUE(Tensor::allclose(prog.value(out), got, 0.0, 0.0))
            << "re-execution changed results at node " << out;
    }
}

TEST(Jit, MatchesInterpreterOnEveryGradientOutputOfLossFixture) {
    auto f = test_fixtures::make_loss_fixture();
    auto grads = grad(f.g, f.loss, {f.W, f.x, f.y});
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };
    jit::JitProgram prog(f.g);
    prog.execute(inputs);
    for (NodeId gid : grads) {
        Tensor expected = eval(f.g, inputs, gid);
        EXPECT_TRUE(Tensor::allclose(expected, prog.value(gid), 1e-9, 1e-6))
            << "jit gradient mismatch at node " << gid;
    }
}

TEST(Jit, RejectsPhase9OpsAtCompileTimeWithClearError) {
    Graph g;
    NodeId x = g.input({1, 4, 4});
    NodeId p = g.maxpool(x, 2, 2, 2);
    (void)p;
    try {
        jit::JitProgram prog(g);
        FAIL() << "expected JitProgram construction to throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("MaxPool"), std::string::npos)
            << "error should name the unsupported op, got: " << e.what();
    }
}

TEST(Jit, RejectsWrongInputCountAndShape) {
    auto f = test_fixtures::make_loss_fixture();
    jit::JitProgram prog(f.g);
    EXPECT_THROW(prog.execute({}), std::invalid_argument);
    EXPECT_THROW(prog.execute({Tensor::zeros({3, 3}),
                               Tensor::zeros({2, 1}),
                               Tensor::zeros({2, 1})}),
                 std::invalid_argument);
    EXPECT_NO_THROW(prog.execute({Tensor::zeros({2, 2}),
                                  Tensor::zeros({2, 1}),
                                  Tensor::zeros({2, 1})}));
}

TEST(Jit, ProgramReuseAcrossDifferentInputs) {


    Graph g;
    NodeId a = g.input({2});
    NodeId s = g.sum(a);
    jit::JitProgram prog(g);

    prog.execute({Tensor::from_vec({2}, {1.0, 2.0})});
    double first = prog.value(s).item();
    prog.execute({Tensor::from_vec({2}, {10.0, 20.0})});
    double second = prog.value(s).item();
    EXPECT_NEAR(first, 3.0, 1e-12);
    EXPECT_NEAR(second, 30.0, 1e-12);
}


TEST(Jit, SpeedupVsInterpreterOnMatmulChainPrinted) {
    constexpr int kDim = 24;
    constexpr int kDepth = 40;
    constexpr int kIters = 30;

    Graph g;
    NodeId cur = g.input({kDim, kDim});
    NodeId w = g.constant(0.999, {kDim, kDim});
    for (int i = 0; i < kDepth; ++i) cur = g.matmul(cur, w);
    NodeId out = g.sum(cur);

    std::vector<Tensor> inputs = {Tensor::from_vec(
        {kDim, kDim}, std::vector<double>(kDim * kDim, 0.001))};

    Tensor want = eval(g, inputs, out);
    jit::JitProgram prog(g);
    prog.execute(inputs);
    ASSERT_TRUE(Tensor::allclose(want, prog.value(out), 1e-7, 1e-7));

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    for (int i = 0; i < kIters; ++i) (void)eval(g, inputs, out);
    auto t1 = clock::now();
    for (int i = 0; i < kIters; ++i) prog.execute(inputs);
    auto t2 = clock::now();

    double interp_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / kIters;
    double jit_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count() / kIters;
    std::cout << std::fixed << std::setprecision(3)
              << "[Jit speedup] matmul chain depth=" << kDepth
              << " dim=" << kDim << ": interp=" << interp_ms << "ms"
              << " jit=" << jit_ms << "ms"
              << " speedup=" << (interp_ms / jit_ms) << "x\n";
    EXPECT_GT(interp_ms / jit_ms, 0.0);
}
