#include <gtest/gtest.h>
#include "minijax/frontend.hpp"
#include "minijax/interp.hpp"

using namespace minijax;
using namespace minijax::frontend;

TEST(Frontend, LexesTokensAndSkipsComments) {
    auto toks = lex("# only a comment\nlet x1 = 1.5; # trailing\n");
    ASSERT_GE(toks.size(), 6u);
    EXPECT_EQ(toks[0].kind, TokKind::KwLet);
    EXPECT_EQ(toks[0].line, 2);
    EXPECT_EQ(toks[1].kind, TokKind::Ident);
    EXPECT_EQ(toks[1].text, "x1");
    EXPECT_EQ(toks[3].kind, TokKind::Number);
    EXPECT_NEAR(toks[3].value, 1.5, 1e-12);
    EXPECT_EQ(toks.back().kind, TokKind::End);
}

TEST(Frontend, LexRejectsIllegalCharacters) {
    EXPECT_THROW(lex("let $ = 1;"), std::invalid_argument);
}

TEST(Frontend, ParsesPrecedenceAndCallsIntoCorrectGraph) {

    std::string src = R"(
        let x = input([2, 2]);
        let y = 2 + 3 * relu(x)';
        output y;
    )";
    Module m = parse(src);
    ASSERT_EQ(m.stmts.size(), 3u);

    LowerResult r = lower(m);


    std::vector<Tensor> inputs = {Tensor::from_vec({2, 2}, {1.0, -1.0, 2.0, 0.0})};
    Tensor got = eval(r.g, inputs, r.output);

    Tensor want = Tensor::from_vec({2, 2}, {5.0, 8.0, 2.0, 2.0});
    EXPECT_TRUE(Tensor::allclose(want, got));
}

TEST(Frontend, LossProgramRunsFromSourceTextAndMatchesHandComputed) {
    std::string src = R"(
        # tiny loss: sum(relu(W @ x) - y)
        let W = input([2, 2]);
        let x = input([2, 1]);
        let y = input([2, 1]);
        let pred = relu(W @ x);
        let diff = pred - y;
        let loss = sum(diff);
        output loss;
    )";
    LowerResult r = compile_source(src);
    std::vector<Tensor> inputs = {
        Tensor::from_vec({2, 2}, {0.5, -0.3, 1.2, 0.7}),
        Tensor::from_vec({2, 1}, {1.0, -2.0}),
        Tensor::from_vec({2, 1}, {0.1, 0.2}),
    };


    Tensor loss = eval(r.g, inputs, r.output);
    EXPECT_NEAR(loss.item(), 0.8, 1e-12);


    EXPECT_TRUE(r.bindings.count("W") && r.bindings.count("loss"));
}

TEST(Frontend, ShapeErrorsAreReadableWithLineInfo) {
    std::string src = R"(# line 1 is a comment
let W = input([2, 2]);
let v = input([5, 2]);
output W @ v;
)";
    try {
        LowerResult r = compile_source(src);
        (void)r;
        FAIL() << "expected shape error";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("line 4"), std::string::npos) << "message was: " << msg;
        EXPECT_NE(msg.find("matmul"), std::string::npos) << "message was: " << msg;
        EXPECT_NE(msg.find("(2 vs 5"), std::string::npos)
            << "message should name the mismatched inner dims: " << msg;
    }
}

TEST(Frontend, SemanticErrorsAreReadableToo) {
    EXPECT_THROW(compile_source("output 1 + missing_var;"), std::invalid_argument);
    EXPECT_THROW(compile_source("let a = relu(1, 2); output a;"), std::invalid_argument);
    EXPECT_THROW(compile_source("let a = frobnicate(a); output a;"), std::invalid_argument);
    EXPECT_THROW(compile_source("let a = input([2]);"), std::invalid_argument);
    EXPECT_THROW(
        compile_source("let a = input([2]); let a = abs(a); output a;"),
        std::invalid_argument);
}
