#include "minijax/frontend.hpp"

#include <functional>

namespace minijax::frontend {

namespace {

[[noreturn]] void parse_error(int line, const std::string& msg) {
    throw std::invalid_argument("parse error at line " + std::to_string(line) + ": " + msg);
}

class Parser {
public:
    explicit Parser(std::vector<Token> toks) : toks_(std::move(toks)) {}

    Module parse_module() {
        Module m;
        while (peek().kind != TokKind::End) {
            m.stmts.push_back(parse_stmt());
        }
        return m;
    }

private:
    const std::vector<Token> toks_;
    size_t pos_ = 0;

    const Token& peek(size_t off = 0) const {
        size_t p = pos_ + off;
        return p < toks_.size() ? toks_[p] : toks_.back();
    }
    const Token& advance() { return toks_[std::min(pos_++, toks_.size() - 1)]; }

    [[noreturn]] void unexpected(const std::string& context) {
        const Token& t = peek();
        parse_error(t.line, "unexpected " + std::string(tok_kind_name(t.kind)) +
                                 (t.text.empty() ? "" : " '" + t.text + "'") + " in " + context);
    }

    bool eat(TokKind k) {
        if (peek().kind == k) { advance(); return true; }
        return false;
    }

    void expect(TokKind k, const char* context) {
        if (!eat(k)) {
            const Token& t = peek();
            parse_error(t.line, "expected " + std::string(tok_kind_name(k)) + " in " + context +
                                     ", found " + tok_kind_name(t.kind));
        }
    }

    Stmt parse_stmt() {
        Stmt s;
        s.line = peek().line;
        if (eat(TokKind::KwLet)) {
            s.is_output = false;
            const Token& name = peek();
            if (name.kind != TokKind::Ident) unexpected("let target");
            s.name = advance().text;
            expect(TokKind::Equal, "let binding");
            s.value = parse_expr();
            expect(TokKind::Semi, "end of let statement");
        } else if (eat(TokKind::KwOutput)) {
            s.is_output = true;
            s.value = parse_expr();
            expect(TokKind::Semi, "end of output statement");
        } else {
            unexpected("statement (expected 'let' or 'output')");
        }
        return s;
    }


    ExprPtr parse_expr() {
        ExprPtr lhs = parse_mult();
        while (peek().kind == TokKind::Plus || peek().kind == TokKind::Minus) {
            Token op = advance();
            ExprPtr rhs = parse_mult();
            auto node = std::make_unique<BinOp>();
            node->op = op.kind == TokKind::Plus ? '+' : '-';
            node->line = op.line;
            node->lhs = std::move(lhs);
            node->rhs = std::move(rhs);
            lhs = std::move(node);
        }
        return lhs;
    }


    ExprPtr parse_mult() {
        ExprPtr lhs = parse_unary();
        while (peek().kind == TokKind::Star || peek().kind == TokKind::Slash ||
               peek().kind == TokKind::At) {
            Token op = advance();
            ExprPtr rhs = parse_unary();
            auto node = std::make_unique<BinOp>();
            node->op = op.kind == TokKind::Star ? '*' : op.kind == TokKind::Slash ? '/' : '@';
            node->line = op.line;
            node->lhs = std::move(lhs);
            node->rhs = std::move(rhs);
            lhs = std::move(node);
        }
        return lhs;
    }

    ExprPtr parse_unary() {
        if (peek().kind == TokKind::Minus) {
            Token op = advance();
            auto node = std::make_unique<UnaryOp>();
            node->op = '-';
            node->line = op.line;
            node->operand = parse_unary();
            return node;
        }
        return parse_postfix();
    }

    ExprPtr parse_postfix() {
        ExprPtr e = parse_primary();
        while (peek().kind == TokKind::Apostrophe) {
            Token op = advance();
            auto node = std::make_unique<PostfixOp>();
            node->op = '\'';
            node->line = op.line;
            node->operand = std::move(e);
            e = std::move(node);
        }
        return e;
    }

    ExprPtr parse_primary() {
        const Token& t = peek();
        int line = t.line;
        switch (t.kind) {
            case TokKind::Number: {
                advance();
                auto n = std::make_unique<NumLit>();
                n->line = line;
                n->value = t.value;
                return n;
            }
            case TokKind::Ident: {
                advance();
                if (peek().kind == TokKind::LParen) {
                    advance();
                    auto call = std::make_unique<CallExpr>();
                    call->line = line;
                    call->callee = t.text;
                    if (peek().kind != TokKind::RParen) {
                        call->args.push_back(parse_expr());
                        while (eat(TokKind::Comma)) call->args.push_back(parse_expr());
                    }
                    expect(TokKind::RParen, "call arguments");
                    return call;
                }
                auto v = std::make_unique<VarRef>();
                v->line = line;
                v->name = t.text;
                return v;
            }
            case TokKind::LParen: {
                advance();
                ExprPtr inner = parse_expr();
                expect(TokKind::RParen, "parenthesized expression");
                return inner;
            }
            case TokKind::LBracket:
                return parse_shape_lit();
            default:
                unexpected("expression");
        }
    }

    ExprPtr parse_shape_lit() {
        const Token& open = peek();
        expect(TokKind::LBracket, "shape literal");
        auto sh = std::make_unique<ShapeLit>();
        sh->line = open.line;
        while (peek().kind == TokKind::Number) {
            double d = advance().value;
            if (d < 1.0 || d != static_cast<double>(static_cast<size_t>(d))) {
                parse_error(open.line, "shape dimensions must be positive integers");
            }
            sh->dims.push_back(static_cast<size_t>(d));
            if (!eat(TokKind::Comma)) break;
        }
        expect(TokKind::RBracket, "shape literal");
        return sh;
    }
};

}

Module parse(const std::string& src) {
    Parser p(lex(src));
    return p.parse_module();
}

}
