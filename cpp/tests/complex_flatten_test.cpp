#include <iostream>
#include <cmath>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

int main() {
    try {
        using namespace minijax;

        // Build a vectorized expression mixing associative adds and constants
        Graph g;
        // inputs: a,b,c are vectors of size 2
        Shape v2 = {2};
        auto a = g.input(v2);
        auto b = g.input(v2);
        auto c = g.input(v2);

        auto k1 = g.constant(1.0, v2); // vector constant
        auto k4 = g.constant(4.0, v2);

        // structure: root = (a + (b + k1)) + (c + k4)
        auto t1 = g.add(b, k1);
        auto t2 = g.add(c, k4);
        auto left = g.add(a, t1);
        auto root = g.add(left, t2);
        auto s = g.sum(root);

        // inputs values
        Tensor va(v2, {1.0, 2.0});
        Tensor vb(v2, {3.0, 4.0});
        Tensor vc(v2, {5.0, 6.0});
        auto before = eval(g, {va, vb, vc});
        double vbv = before[s].scalar();

        // run all optimizer steps
        rebuild_with_assoc(g);
        optimize_until_fixedpoint(g);
        extract_min_cost(g);

        // find sum node after transforms
        size_t sum_idx = 0;
        for (size_t i = g.nodes.size(); i-- > 0;) if (g.nodes[i].op == OpKind::Sum) { sum_idx = i; break; }
        auto after = eval(g, {va, vb, vc});
        double vav = after[sum_idx].scalar();

        if (std::abs(vbv - vav) > 1e-12) {
            std::cerr << "complex flatten value mismatch: before=" << vbv << " after=" << vav << "\n";
            return 1;
        }

        std::cout << "complex flatten ok\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "complex flatten test error: " << e.what() << std::endl;
        return 1;
    }
}
