#pragma once


#include <string>
#include <vector>
#include "minijax/ir.hpp"

namespace minijax {


std::string to_dot(const Graph& g);


std::string to_dot(const Graph& g, const std::vector<NodeId>& highlight);

}
