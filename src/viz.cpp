#include "minijax/viz.hpp"

#include <algorithm>
#include <sstream>

namespace minijax {

std::string to_dot(const Graph& g, const std::vector<NodeId>& highlight) {
    std::ostringstream os;
    os << "digraph minijax {\n";
    os << "  rankdir=\"BT\"; // data flows bottom-up like the arena order\n";

    std::vector<bool> hot(g.size(), false);
    for (NodeId id : highlight) {
        if (id < g.size()) hot[id] = true;
    }

    for (NodeId i = 0; i < g.size(); ++i) {
        const Node& n = g.node(i);
        os << "  n" << i << " [label=\"" << i << ": " << op_kind_name(n.op) << "\\n[";
        for (size_t d = 0; d < n.shape.size(); ++d) {
            if (d) os << ",";
            os << n.shape[d];
        }
        os << "]\"";
        if (n.op == OpKind::Input) os << ", shape=box";
        if (hot[i]) os << ", style=filled, fillcolor=\"lightcoral\"";
        else if (n.op == OpKind::Input) os << ", style=filled, fillcolor=\"lightblue\"";
        else if (n.op == OpKind::Const) os << ", style=filled, fillcolor=\"lightgray\"";
        os << "];\n";
    }
    for (NodeId i = 0; i < g.size(); ++i) {
        for (NodeId in : g.node(i).inputs) {
            os << "  n" << in << " -> n" << i << ";\n";
        }
    }
    os << "}\n";
    return os.str();
}

std::string to_dot(const Graph& g) {
    return to_dot(g, {});
}

}
