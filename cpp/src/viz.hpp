#pragma once

#include <sstream>
#include <string>

#include "ir.hpp"

namespace minijax {

inline std::string dump_graph_dot(const Graph& g) {
    std::ostringstream oss;
    oss << "digraph G {\n";
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        oss << "  n" << i << " [label=\"node_" << i << "\"];\n";
        for (size_t in : g.nodes[i].inputs) {
            oss << "  n" << in << " -> n" << i << "\n";
        }
    }
    oss << "}\n";
    return oss.str();
}

}  // namespace minijax
