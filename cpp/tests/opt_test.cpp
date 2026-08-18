#include <iostream>
#include <stdexcept>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

int main() {
    try {
        minijax::Graph g;
        // Test 1: simple const fold
        {
            auto a = g.constant(2.0, {});
            auto b = g.constant(3.0, {});
            auto c = g.add(a, b);
            auto s = g.sum(c);

            auto before = minijax::eval(g, {});
            double val_before = before[s].scalar();

            minijax::optimize(g);

            auto after = minijax::eval(g, {});
            double val_after = after[s].scalar();

            if (std::abs(val_before - val_after) > 1e-12) {
                std::cerr << "opt test failed: values differ\n";
                return 1;
            }
        }

        // Test 2: x + 0 -> x
        {
            minijax::Graph g2;
            auto x = g2.input({});
            auto z = g2.constant(0.0, {});
            auto a = g2.add(x, z);
            auto s = g2.sum(a);
            auto before = minijax::eval(g2, {minijax::Tensor::scalar(5.0)});
            double v_before = before[s].scalar();
            minijax::optimize(g2);
            auto after = minijax::eval(g2, {minijax::Tensor::scalar(5.0)});
            double v_after = after[s].scalar();
            if (std::abs(v_before - v_after) > 1e-12) { std::cerr << "opt test x+0 failed\n"; return 1; }
        }

        // Test 3: x * 1 -> x, x * 0 -> 0
        {
            minijax::Graph g3;
            auto x = g3.input({});
            auto one = g3.constant(1.0, {});
            auto zero = g3.constant(0.0, {});
            auto m1 = g3.mul(x, one);
            auto m0 = g3.mul(x, zero);
            auto s1 = g3.sum(m1);
            auto s0 = g3.sum(m0);
            auto before1 = minijax::eval(g3, {minijax::Tensor::scalar(7.0)});
            double b1 = before1[s1].scalar();
            double b0 = before1[s0].scalar();
            minijax::optimize(g3);
            auto after1 = minijax::eval(g3, {minijax::Tensor::scalar(7.0)});
            double a1 = after1[s1].scalar();
            double a0 = after1[s0].scalar();
            if (std::abs(b1 - a1) > 1e-12) { std::cerr << "opt test x*1 failed\n"; return 1; }
            if (std::abs(a0 - 0.0) > 1e-12) { std::cerr << "opt test x*0 failed\n"; return 1; }
        }

        // Test 4: fixed-point driver idempotence
        {
            minijax::Graph g4;
            auto a = g4.constant(1.0, {});
            auto b = g4.constant(2.0, {});
            auto c = g4.add(a, b);
            auto d = g4.add(c, g4.constant(3.0, {}));
            auto s = g4.sum(d);
            minijax::optimize_until_fixedpoint(g4);
            auto v1 = minijax::eval(g4, {});
            minijax::optimize_until_fixedpoint(g4);
            auto v2 = minijax::eval(g4, {});
            if (v1[s].scalar() != v2[s].scalar()) { std::cerr << "fixed-point failed\n"; return 1; }
        }

        // Test 5: (2 + (x + 3)) -> (x + 5) value preserved
        {
            minijax::Graph g5;
            auto c2 = g5.constant(2.0, {});
            auto x = g5.input({});
            auto c3 = g5.constant(3.0, {});
            auto inner = g5.add(x, c3); // index earlier than root
            auto root = g5.add(c2, inner);
            auto s = g5.sum(root);
            double xv = 4.0;
            auto before = minijax::eval(g5, {minijax::Tensor::scalar(xv)});
            double vb = before[s].scalar();
            std::cerr << "graph before:\n" << minijax::graph_fingerprint(g5);
            auto dump = [&](const minijax::Graph& gg){
                for (size_t i = 0; i < gg.nodes.size(); ++i) {
                    const auto& n = gg.nodes[i];
                    std::cerr << i << ": op=" << static_cast<int>(n.op) << " inputs={";
                    for (size_t j = 0; j < n.inputs.size(); ++j) { if (j) std::cerr<<","; std::cerr<<n.inputs[j]; }
                    std::cerr << "} const=" << n.const_value << " shape_sz=" << n.shape.size() << "\n";
                }
            };
            dump(g5);
            minijax::optimize_all(g5);
            std::cerr << "graph after:\n";
            dump(g5);
            // node indices may shift after rebuild; locate Sum node
            size_t new_s = g5.nodes.size() - 1;
            if (g5.nodes[new_s].op != minijax::OpKind::Sum) {
                for (size_t i = g5.nodes.size(); i-- > 0;) if (g5.nodes[i].op == minijax::OpKind::Sum) { new_s = i; break; }
            }
            auto after = minijax::eval(g5, {minijax::Tensor::scalar(xv)});
            double va = after[new_s].scalar();
            if (std::abs(vb - va) > 1e-12) {
                std::cerr << "assoc rewrite value mismatch: before=" << vb << " after=" << va << "\n";
                return 1;
            }
        }

        std::cout << "opt ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "opt test failed: " << ex.what() << std::endl;
        return 1;
    }
}
