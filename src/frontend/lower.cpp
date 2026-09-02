#include "minijax/frontend.hpp"

#include <functional>

namespace minijax::frontend {

namespace {

class Lowerer {
public:
    LowerResult run(const Module& m) {
        for (const auto& stmt : m.stmts) {
            NodeId v = eval_expr(*stmt.value);
            if (stmt.is_output) {
                result_.outputs.push_back(v);
                result_.output = v;
            } else {
                if (result_.bindings.count(stmt.name)) {
                    throw std::invalid_argument("line " + std::to_string(stmt.line) +
                                                 ": duplicate binding for '" + stmt.name + "'");
                }
                result_.bindings.emplace(stmt.name, v);
            }
        }
        if (result_.outputs.empty()) {
            throw std::invalid_argument("program has no 'output' statement");
        }
        return std::move(result_);
    }

private:
    LowerResult result_;

    [[noreturn]] static void line_error(int line, const std::string& msg) {


        throw std::invalid_argument("line " + std::to_string(line) + ": " + msg);
    }

    static const ShapeLit& expect_shape(const Expr& e, const std::string& context) {
        const auto* sh = dynamic_cast<const ShapeLit*>(&e);
        if (!sh) throw std::invalid_argument(context + " expects a shape literal like [2, 3]");
        return *sh;
    }

    NodeId eval_expr(const Expr& e) {


        auto already_annotated = [](const std::string& m) {
            return m.rfind("line ", 0) == 0;
        };
        try {
            return eval_inner(e);
        } catch (const std::invalid_argument& ex) {
            if (already_annotated(ex.what())) throw;
            line_error(e.line, ex.what());
        } catch (const std::runtime_error& ex) {
            if (already_annotated(ex.what())) throw;
            line_error(e.line, ex.what());
        }

    }

    NodeId eval_inner(const Expr& e) {
        if (const auto* num = dynamic_cast<const NumLit*>(&e)) {
            return result_.g.constant(num->value);
        }
        if (const auto* var = dynamic_cast<const VarRef*>(&e)) {
            auto it = result_.bindings.find(var->name);
            if (it == result_.bindings.end()) {
                throw std::runtime_error("unknown identifier '" + var->name +
                                          "' (use-before-let?)");
            }
            return it->second;
        }
        if (const auto* un = dynamic_cast<const UnaryOp*>(&e)) {
            NodeId x = eval_expr(*un->operand);
            return result_.g.neg(x);
        }
        if (const auto* post = dynamic_cast<const PostfixOp*>(&e)) {
            NodeId x = eval_expr(*post->operand);
            return result_.g.transpose(x);
        }
        if (const auto* bin = dynamic_cast<const BinOp*>(&e)) {
            NodeId lhs = eval_expr(*bin->lhs);
            NodeId rhs = eval_expr(*bin->rhs);
            switch (bin->op) {
                case '+': return result_.g.add(lhs, rhs);
                case '-': return result_.g.sub(lhs, rhs);
                case '*': return result_.g.mul(lhs, rhs);
                case '/': return result_.g.div(lhs, rhs);
                case '@': return result_.g.matmul(lhs, rhs);
            }
            throw std::runtime_error("internal: unknown binary op");
        }
        if (const auto* call = dynamic_cast<const CallExpr*>(&e)) {
            return eval_call(*call);
        }
        throw std::runtime_error("internal: unknown expression kind");
    }

    size_t arg_count(const CallExpr& c, size_t n) {
        if (c.args.size() != n) {
            throw std::runtime_error("'" + c.callee + "' expects " + std::to_string(n) +
                                      " argument(s), got " + std::to_string(c.args.size()));
        }
        return 0;
    }

    std::vector<size_t> shape_arg(const CallExpr& c, size_t idx) {
        return expect_shape(*c.args[idx], "'" + c.callee + "'").dims;
    }

    NodeId eval_call(const CallExpr& c) {
        Graph& g = result_.g;
        const std::string& f = c.callee;

        if (f == "input") {
            if (c.args.size() != 1)
                throw std::runtime_error("'input' expects 1 argument (a shape), got " +
                                          std::to_string(c.args.size()));
            return g.input(shape_arg(c, 0));
        }
        if (f == "sum_axis") {
            arg_count(c, 2);
            const auto* axis_lit = dynamic_cast<const NumLit*>(c.args[1].get());
            if (!axis_lit || axis_lit->value < 0 ||
                axis_lit->value != static_cast<double>(static_cast<size_t>(axis_lit->value))) {
                throw std::runtime_error("'sum_axis' second argument must be a non-negative integer literal");
            }
            return g.sum_axis(eval_expr(*c.args[0]), static_cast<size_t>(axis_lit->value));
        }
        if (f == "reshape") { arg_count(c, 2); return g.reshape(eval_expr(*c.args[0]), shape_arg(c, 1)); }
        if (f == "broadcast") { arg_count(c, 2); return g.broadcast_to(eval_expr(*c.args[0]), shape_arg(c, 1)); }

        if (c.args.size() != 1)
            throw std::runtime_error("'" + f + "' expects 1 argument, got " +
                                      std::to_string(c.args.size()));
        NodeId x = eval_expr(*c.args[0]);
        if (f == "relu") return g.relu(x);
        if (f == "step") return g.step(x);
        if (f == "tanh") return g.tanh(x);
        if (f == "sigmoid") return g.sigmoid(x);
        if (f == "exp") return g.exp(x);
        if (f == "log") return g.log(x);
        if (f == "sqrt") return g.sqrt(x);
        if (f == "abs") return g.abs(x);
        if (f == "sum") return g.sum(x);
        if (f == "transpose") return g.transpose(x);

        throw std::runtime_error("unknown function '" + f + "'");
    }
};

}

LowerResult lower(const Module& m) {
    return Lowerer().run(m);
}

LowerResult compile_source(const std::string& src) {
    return lower(parse(src));
}

}
