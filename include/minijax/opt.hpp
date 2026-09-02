#pragma once


#include "minijax/ir.hpp"

namespace minijax {


std::pair<Graph, NodeId> optimize(const Graph& g, NodeId output);


std::pair<Graph, NodeId> optimize_sound(const Graph& g, NodeId output);

}
