#include <iostream>
#include <cmath>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

int main() {
    try {
        using namespace minijax;

        // Test A: (2 + (x + 3) + (y + 4)) -> value preserved and constant combined
        {
            Graph g;
            auto c2 = g.constant(2.0, {});
            auto x = g.input({});
            auto c3 = g.constant(3.0, {});
            auto inner = g.add(x, c3);
            auto y = g.input({});
            auto c4 = g.constant(4.0, {});
            auto right = g.add(y, c4);
            auto t1 = g.add(c2, inner);
            auto root = g.add(t1, right);
            auto s = g.sum(root);

            double xv = 5.0, yv = 7.0;
            auto before = eval(g, {Tensor::scalar(xv), Tensor::scalar(yv)});
            double vb = before[s].scalar();

            optimize_all(g);

            // value after
            size_t sum_idx = 0;
            for (size_t i = g.nodes.size(); i-- > 0;) if (g.nodes[i].op == OpKind::Sum) { sum_idx = i; break; }
            auto after = eval(g, {Tensor::scalar(xv), Tensor::scalar(yv)});
            double va = after[sum_idx].scalar();
            if (std::abs(vb - va) > 1e-12) { std::cerr << "flatten value mismatch\n"; return 1; }

            // check combined constant 9 (2+3+4) exists
            bool found9 = false;
            for (const auto &n : g.nodes) if (n.op == OpKind::Const && std::abs(n.const_value - 9.0) < 1e-12) found9 = true;
            if (!found9) { std::cerr << "combined constant 9 not found\n"; return 1; }
        }

        // Test B: canonical ordering: (x + y) and (y + x) should fingerprint equal after normalize
        {
            Graph g1;
            auto x1 = g1.input({});
            auto y1 = g1.input({});
            auto a1 = g1.add(x1, y1);
            auto s1 = g1.sum(a1);

            Graph g2;
            auto x2 = g2.input({});
            auto y2 = g2.input({});
            auto a2 = g2.add(y2, x2);
            auto s2 = g2.sum(a2);

            optimize_all(g1);
            optimize_all(g2);

            auto f1 = graph_fingerprint(g1);
            auto f2 = graph_fingerprint(g2);
            if (f1 != f2) {
                std::cerr << "canonical ordering mismatch\n";
                std::cerr << "fingerprint g1:\n" << f1 << "\n";
                std::cerr << "fingerprint g2:\n" << f2 << "\n";
                auto dump = [&](const Graph &g){
                    for (size_t i = 0; i < g.nodes.size(); ++i) {
                        const auto &n = g.nodes[i];
                        std::cerr << i << ": op=" << static_cast<int>(n.op) << " inputs={";
                        for (size_t j = 0; j < n.inputs.size(); ++j) { if (j) std::cerr<<","; std::cerr<<n.inputs[j]; }
                        std::cerr << "} const=" << n.const_value << "\n";
                    }
                };
                std::cerr << "graph g1:\n"; dump(g1);
                std::cerr << "graph g2:\n"; dump(g2);
                return 1;
            }
        }

        std::cout << "flatten tests ok\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "flatten test error: " << e.what() << std::endl;
        return 1;
    }
}
