#include <iostream>
#include <cmath>

#include "interp.hpp"
#include "ir.hpp"
#include "opt.hpp"
#include "tensor.hpp"

int main() {
    try {
        using namespace minijax;

        // Build a tree with nested expressions: ((a+b)+(c+(d+e))) etc.
        Graph g;
        auto a = g.input({});
        auto b = g.input({});
        auto c = g.input({});
        auto d = g.input({});
        auto e = g.input({});
        auto n1 = g.add(a, b);
        auto n2 = g.add(d, e);
        auto n3 = g.add(c, n2);
        auto root = g.add(n1, n3);
        auto s = g.sum(root);

        // evaluate before
        std::vector<minijax::Tensor> inputs = {minijax::Tensor::scalar(1.0), minijax::Tensor::scalar(2.0), minijax::Tensor::scalar(3.0), minijax::Tensor::scalar(4.0), minijax::Tensor::scalar(5.0)};
        auto before_vals = minijax::eval(g, inputs);
        double vb = before_vals[s].scalar();

        rebuild_with_assoc(g);
        optimize_until_fixedpoint(g);
        extract_min_cost(g);

        // find sum node after transforms
        size_t sum_idx = 0;
        for (size_t i = g.nodes.size(); i-- > 0;) if (g.nodes[i].op == minijax::OpKind::Sum) { sum_idx = i; break; }
        auto after_vals = minijax::eval(g, inputs);
        double va = after_vals[sum_idx].scalar();
        if (std::abs(vb - va) > 1e-12) {
            std::cerr << "extractor value mismatch: before=" << vb << " after=" << va << "\n";
            return 1;
        }

        std::cout << "extract test ok\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "extract test error: " << e.what() << std::endl;
        return 1;
    }
}
