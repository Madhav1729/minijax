#pragma once


#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "minijax/ir.hpp"

namespace minijax::frontend {


enum class TokKind {
    Ident, Number, KwLet, KwOutput,
    LParen, RParen, LBracket, RBracket, Comma, Semi, Equal,
    Plus, Minus, Star, Slash, At, Apostrophe, End,
};

struct Token {
    TokKind kind;
    std::string text;
    double value = 0.0;
    int line = 0;
};


std::vector<Token> lex(const std::string& src);

const char* tok_kind_name(TokKind k);


struct Expr {
    int line = 0;
    virtual ~Expr() = default;
};
using ExprPtr = std::unique_ptr<Expr>;

struct NumLit : Expr { double value = 0.0; };
struct VarRef : Expr { std::string name; };
struct ShapeLit : Expr { std::vector<size_t> dims; };
struct UnaryOp : Expr { char op = '-'; ExprPtr operand; };
struct PostfixOp : Expr { char op = '\''; ExprPtr operand; };
struct BinOp : Expr { char op = '+'; ExprPtr lhs, rhs; };
struct CallExpr : Expr { std::string callee; std::vector<ExprPtr> args; };

struct Stmt {
    int line = 0;
    bool is_output = false;
    std::string name;
    ExprPtr value;
};

struct Module {
    std::vector<Stmt> stmts;
};


Module parse(const std::string& src);


struct LowerResult {
    Graph g;
    NodeId output = 0;
    std::vector<NodeId> outputs;
    std::unordered_map<std::string, NodeId> bindings;
};


LowerResult lower(const Module& m);


LowerResult compile_source(const std::string& src);

}
