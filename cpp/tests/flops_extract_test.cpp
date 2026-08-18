#include <iostream>
#include <cmath>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

int main() {
    try {
        using namespace minijax;

        // simple test: elementwise adds and muls with different shapes
        Graph g;
        Shape s1 = {2,2};
        Shape s2 = {2};
        auto A = g.input(s1);
        auto B = g.input(s1);
        auto c = g.constant(2.0, s2);
        auto cb = g.broadcast(c, s1);
        auto t = g.add(A, B);
        auto root = g.add(t, cb);
        auto out = g.sum(root);

        Tensor a(s1, {1,2,3,4});
        Tensor b(s1, {5,6,7,8});
        auto before = eval(g, {a,b});
        double vb = before[out].scalar();

        rebuild_with_assoc(g);
        optimize_until_fixedpoint(g);
        extract_by_flops(g);

        size_t sum_idx = 0;
        for (size_t i = g.nodes.size(); i-- > 0;) if (g.nodes[i].op == OpKind::Sum) { sum_idx = i; break; }
        auto after = eval(g, {a,b});
        double va = after[sum_idx].scalar();
        if (std::abs(vb - va) > 1e-12) { std::cerr << "flops extract mismatch\n"; return 1; }

        std::cout << "flops extract ok\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "flops extract test error: " << e.what() << std::endl;
        return 1;
    }
}
