#include "minijax/opt.hpp"
#include "minijax/egraph.hpp"

namespace minijax {

std::pair<Graph, NodeId> optimize(const Graph& g, NodeId output) {
    auto [eg, node_to_eclass] = to_egraph(g);
    saturate(eg);
    NodeId new_output;
    Graph out = extract(eg, g, node_to_eclass, node_to_eclass[output], new_output);
    return {std::move(out), new_output};
}

std::pair<Graph, NodeId> optimize_sound(const Graph& g, NodeId output) {


    auto [eg, node_to_eclass] = to_egraph(g);
    saturate_ex(eg, true);
    NodeId new_output;
    Graph out = extract(eg, g, node_to_eclass, node_to_eclass[output], new_output);
    return {std::move(out), new_output};
}

}
