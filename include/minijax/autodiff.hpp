#pragma once


#include <vector>
#include "minijax/ir.hpp"

namespace minijax {


std::vector<NodeId> grad(Graph& g, NodeId output, const std::vector<NodeId>& wrt);

}
