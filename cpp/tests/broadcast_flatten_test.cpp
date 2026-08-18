#include <iostream>
#include <cmath>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

int main() {
    try {
        using namespace minijax;

        // Test: folding should preserve semantics when Broadcast/Reshape are present
        Graph g;
        Shape v2 = {2};
        auto a = g.input(v2);            // vector
        auto s = g.constant(3.0, {});    // scalar
        auto sb = g.broadcast(s, v2);   // broadcast scalar to vector
        auto b = g.input(v2);            // another vector

        // root = (a + sb) + b
        auto t1 = g.add(a, sb);
        auto root = g.add(t1, b);
        auto out = g.sum(root);

        Tensor va(v2, {1.0, 2.0});
        Tensor vb(v2, {3.0, 4.0});
        auto before = eval(g, {va, vb});
        double vbefore = before[out].scalar();

        optimize_all(g);

        // locate sum node
        size_t sum_idx = 0;
        for (size_t i = g.nodes.size(); i-- > 0;) if (g.nodes[i].op == OpKind::Sum) { sum_idx = i; break; }
        auto after = eval(g, {va, vb});
        double vafter = after[sum_idx].scalar();

        if (std::abs(vbefore - vafter) > 1e-12) {
            std::cerr << "broadcast flatten mismatch: before=" << vbefore << " after=" << vafter << "\n";
            return 1;
        }

        std::cout << "broadcast flatten ok\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "broadcast test error: " << e.what() << std::endl;
        return 1;
    }
}
