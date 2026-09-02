#pragma once


#include <vector>
#include <unordered_map>
#include <cstddef>
#include "minijax/ir.hpp"

namespace minijax {

using EClassId = size_t;

struct ENode {
    OpKind op;
    std::vector<EClassId> children;
    std::vector<size_t> shape;

    double const_value = 0.0;
    size_t input_slot = 0;
    size_t axis = 0;
    std::vector<size_t> target_shape;

    bool operator==(const ENode& o) const;
};

struct ENodeHash {
    size_t operator()(const ENode& n) const;
};

class EGraph {
public:

    EClassId add(ENode node);

    EClassId find(EClassId id) const;
    void unite(EClassId a, EClassId b);

    const std::vector<ENode>& members_of(EClassId id) const;

    size_t num_classes() const { return parent_.size(); }

private:
    mutable std::vector<EClassId> parent_;
    std::vector<std::vector<ENode>> members_;
    std::unordered_map<ENode, EClassId, ENodeHash> hashcons_;
};


int saturate(EGraph& eg, int max_iters = 8);


int saturate_ex(EGraph& eg, bool sound_only, int max_iters = 8);


std::pair<EGraph, std::vector<EClassId>> to_egraph(const Graph& g);


Graph extract(const EGraph& eg, const Graph& original, const std::vector<EClassId>& node_to_eclass,
              EClassId root, NodeId& new_output);

}
