#include "minijax/egraph.hpp"
#include <algorithm>
#include <limits>
#include <functional>

namespace minijax {

bool ENode::operator==(const ENode& o) const {
    return op == o.op && children == o.children && shape == o.shape &&
           const_value == o.const_value && input_slot == o.input_slot &&
           axis == o.axis && target_shape == o.target_shape;
}

size_t ENodeHash::operator()(const ENode& n) const {
    size_t h = std::hash<int>{}(static_cast<int>(n.op));
    auto mix = [&h](size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); };
    for (auto c : n.children) mix(std::hash<size_t>{}(c));
    for (auto s : n.shape) mix(std::hash<size_t>{}(s));
    mix(std::hash<double>{}(n.const_value));
    mix(std::hash<size_t>{}(n.input_slot));
    mix(std::hash<size_t>{}(n.axis));
    for (auto s : n.target_shape) mix(std::hash<size_t>{}(s));
    return h;
}

EClassId EGraph::find(EClassId id) const {
    while (parent_[id] != id) {
        parent_[id] = parent_[parent_[id]];
        id = parent_[id];
    }
    return id;
}

void EGraph::unite(EClassId a, EClassId b) {
    EClassId ra = find(a), rb = find(b);
    if (ra == rb) return;
    parent_[ra] = rb;
    for (auto& n : members_[ra]) members_[rb].push_back(n);
    members_[ra].clear();
}

EClassId EGraph::add(ENode node) {
    for (auto& c : node.children) c = find(c);
    auto it = hashcons_.find(node);
    if (it != hashcons_.end()) return find(it->second);
    EClassId id = parent_.size();
    parent_.push_back(id);
    members_.emplace_back();
    members_[id].push_back(node);
    hashcons_.emplace(std::move(node), id);
    return id;
}

const std::vector<ENode>& EGraph::members_of(EClassId id) const {
    return members_[find(id)];
}

std::pair<EGraph, std::vector<EClassId>> to_egraph(const Graph& g) {
    EGraph eg;
    std::vector<EClassId> node_to_eclass(g.size());
    for (NodeId i = 0; i < g.size(); ++i) {
        const Node& n = g.node(i);
        ENode en;
        en.op = n.op;
        en.shape = n.shape;
        en.const_value = n.const_value;
        en.input_slot = n.input_slot;
        en.axis = n.axis;
        en.target_shape = n.target_shape;
        for (NodeId in : n.inputs) en.children.push_back(node_to_eclass[in]);
        node_to_eclass[i] = eg.add(std::move(en));
    }
    return {std::move(eg), std::move(node_to_eclass)};
}

namespace {

bool is_const(const EGraph& eg, EClassId id, double value, const std::vector<size_t>& shape, EClassId* found_id = nullptr) {
    for (const auto& m : eg.members_of(id)) {
        if (m.op == OpKind::Const && m.const_value == value && m.shape == shape) {
            if (found_id) *found_id = eg.find(id);
            return true;
        }
    }
    return false;
}


const ENode* find_member_with_op(const EGraph& eg, EClassId id, OpKind op) {
    for (const auto& m : eg.members_of(id)) {
        if (m.op == op) return &m;
    }
    return nullptr;
}

bool apply_commutativity(EGraph& eg, OpKind kind) {
    bool changed = false;
    size_t n = eg.num_classes();
    for (EClassId id = 0; id < n; ++id) {
        if (eg.find(id) != id) continue;

        std::vector<ENode> members = eg.members_of(id);
        for (const auto& m : members) {
            if (m.op != kind || m.children.size() != 2) continue;
            ENode swapped = m;
            std::swap(swapped.children[0], swapped.children[1]);
            EClassId swapped_id = eg.add(swapped);
            if (eg.find(swapped_id) != eg.find(id)) {
                eg.unite(id, swapped_id);
                changed = true;
            }
        }
    }
    return changed;
}

bool apply_associativity(EGraph& eg, OpKind kind) {
    bool changed = false;
    size_t n = eg.num_classes();
    for (EClassId id = 0; id < n; ++id) {
        if (eg.find(id) != id) continue;
        std::vector<ENode> members = eg.members_of(id);
        for (const auto& m : members) {
            if (m.op != kind || m.children.size() != 2) continue;
            EClassId l = m.children[0], r = m.children[1];


            if (eg.find(l) == eg.find(id) || eg.find(r) == eg.find(id)) continue;


            if (const ENode* lm = find_member_with_op(eg, l, kind)) {
                if (lm->children.size() == 2) {
                    EClassId p = lm->children[0], q = lm->children[1];
                    ENode inner; inner.op = kind; inner.children = {q, r}; inner.shape = m.shape;
                    EClassId inner_id = eg.add(inner);
                    ENode outer; outer.op = kind; outer.children = {p, inner_id}; outer.shape = m.shape;
                    EClassId outer_id = eg.add(outer);
                    if (eg.find(outer_id) != eg.find(id)) { eg.unite(id, outer_id); changed = true; }
                }
            }

            if (const ENode* rm = find_member_with_op(eg, r, kind)) {
                if (rm->children.size() == 2) {
                    EClassId q = rm->children[0], rr = rm->children[1];
                    ENode inner; inner.op = kind; inner.children = {l, q}; inner.shape = m.shape;
                    EClassId inner_id = eg.add(inner);
                    ENode outer; outer.op = kind; outer.children = {inner_id, rr}; outer.shape = m.shape;
                    EClassId outer_id = eg.add(outer);
                    if (eg.find(outer_id) != eg.find(id)) { eg.unite(id, outer_id); changed = true; }
                }
            }
        }
    }
    return changed;
}

bool apply_identities(EGraph& eg, bool skip_mul_zero) {
    bool changed = false;
    size_t n = eg.num_classes();
    for (EClassId id = 0; id < n; ++id) {
        if (eg.find(id) != id) continue;
        std::vector<ENode> members = eg.members_of(id);
        for (const auto& m : members) {
            if (m.children.size() == 2) {
                EClassId a = m.children[0], b = m.children[1];
                if (m.op == OpKind::Add) {
                    if (is_const(eg, b, 0.0, m.shape)) { if (eg.find(a) != eg.find(id)) { eg.unite(id, a); changed = true; } }
                    else if (is_const(eg, a, 0.0, m.shape)) { if (eg.find(b) != eg.find(id)) { eg.unite(id, b); changed = true; } }
                } else if (m.op == OpKind::Sub) {
                    if (is_const(eg, b, 0.0, m.shape)) { if (eg.find(a) != eg.find(id)) { eg.unite(id, a); changed = true; } }
                } else if (m.op == OpKind::Mul) {
                    if (is_const(eg, b, 1.0, m.shape)) { if (eg.find(a) != eg.find(id)) { eg.unite(id, a); changed = true; } }
                    else if (is_const(eg, a, 1.0, m.shape)) { if (eg.find(b) != eg.find(id)) { eg.unite(id, b); changed = true; } }
                    else if (!skip_mul_zero && is_const(eg, b, 0.0, m.shape)) { if (eg.find(b) != eg.find(id)) { eg.unite(id, b); changed = true; } }
                    else if (!skip_mul_zero && is_const(eg, a, 0.0, m.shape)) { if (eg.find(a) != eg.find(id)) { eg.unite(id, a); changed = true; } }
                }
            }
            if (m.op == OpKind::Neg && m.children.size() == 1) {
                if (const ENode* inner = find_member_with_op(eg, m.children[0], OpKind::Neg)) {
                    EClassId grandchild = inner->children[0];
                    if (eg.find(grandchild) != eg.find(id)) { eg.unite(id, grandchild); changed = true; }
                }
            }
        }
    }
    return changed;
}

bool apply_matmul_assoc(EGraph& eg) {
    bool changed = false;
    size_t n = eg.num_classes();
    for (EClassId id = 0; id < n; ++id) {
        if (eg.find(id) != id) continue;
        std::vector<ENode> members = eg.members_of(id);
        for (const auto& m : members) {
            if (m.op != OpKind::MatMul || m.children.size() != 2) continue;
            EClassId l = m.children[0], b = m.children[1];

            if (const ENode* lm = find_member_with_op(eg, l, OpKind::MatMul)) {
                EClassId p = lm->children[0], q = lm->children[1];


                if (!eg.members_of(q).empty() && !eg.members_of(b).empty()) {
                    ENode inner; inner.op = OpKind::MatMul; inner.children = {q, b};
                    inner.shape = {eg.members_of(q)[0].shape[0], eg.members_of(b)[0].shape[1]};
                    EClassId inner_id = eg.add(inner);
                    ENode outer; outer.op = OpKind::MatMul; outer.children = {p, inner_id}; outer.shape = m.shape;
                    EClassId outer_id = eg.add(outer);
                    if (eg.find(outer_id) != eg.find(id)) { eg.unite(id, outer_id); changed = true; }
                }
            }
        }
    }
    return changed;
}

bool apply_const_fold(EGraph& eg) {
    bool changed = false;
    size_t n = eg.num_classes();
    for (EClassId id = 0; id < n; ++id) {
        if (eg.find(id) != id) continue;
        std::vector<ENode> members = eg.members_of(id);
        for (const auto& m : members) {
            if (m.children.size() != 2) continue;
            if (m.op != OpKind::Add && m.op != OpKind::Sub && m.op != OpKind::Mul) continue;
            const ENode* ca = find_member_with_op(eg, m.children[0], OpKind::Const);
            const ENode* cb = find_member_with_op(eg, m.children[1], OpKind::Const);
            if (!ca || !cb) continue;
            double result = m.op == OpKind::Add ? ca->const_value + cb->const_value
                          : m.op == OpKind::Sub ? ca->const_value - cb->const_value
                                                 : ca->const_value * cb->const_value;
            ENode folded; folded.op = OpKind::Const; folded.const_value = result; folded.shape = m.shape;
            EClassId folded_id = eg.add(folded);
            if (eg.find(folded_id) != eg.find(id)) { eg.unite(id, folded_id); changed = true; }
        }
    }
    return changed;
}

}

int saturate(EGraph& eg, int max_iters) {
    return saturate_ex(eg, false, max_iters);
}

int saturate_ex(EGraph& eg, bool sound_only, int max_iters) {


    int iters = 0;
    for (; iters < max_iters; ++iters) {
        bool changed = false;
        changed |= apply_commutativity(eg, OpKind::Add);
        changed |= apply_commutativity(eg, OpKind::Mul);
        if (!sound_only) {
            changed |= apply_associativity(eg, OpKind::Add);
            changed |= apply_associativity(eg, OpKind::Mul);
        }
        changed |= apply_identities(eg, sound_only);
        if (!sound_only) {
            changed |= apply_matmul_assoc(eg);
        }
        changed |= apply_const_fold(eg);
        if (!changed) { ++iters; break; }
    }
    return iters;
}

namespace {

size_t numel(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

size_t self_cost(const ENode& n) {
    switch (n.op) {
        case OpKind::Input:
        case OpKind::Const:
            return 0;
        case OpKind::Add: case OpKind::Sub: case OpKind::Mul: case OpKind::Div:
        case OpKind::Neg: case OpKind::Relu: case OpKind::Step: case OpKind::Abs:
            return numel(n.shape);
        case OpKind::Tanh: case OpKind::Sigmoid: case OpKind::Exp: case OpKind::Log: case OpKind::Sqrt:
            return numel(n.shape) * 4;
        case OpKind::MatMul:
            return numel(n.shape) * 8;
        case OpKind::Sum: case OpKind::SumAxis:
            return numel(n.shape) + 1;
        case OpKind::Transpose: case OpKind::Broadcast: case OpKind::Reshape:
            return numel(n.shape);
        default:
            return numel(n.shape) + 1;
    }
}

using CostMemo = std::unordered_map<EClassId, std::pair<size_t, ENode>>;

const std::pair<size_t, ENode>& best(const EGraph& eg, EClassId root, CostMemo& memo) {
    root = eg.find(root);
    auto it = memo.find(root);
    if (it != memo.end()) return it->second;


    memo.emplace(root, std::make_pair(std::numeric_limits<size_t>::max(), ENode{}));

    size_t best_cost = std::numeric_limits<size_t>::max();
    ENode best_node;
    bool found_any = false;
    for (const auto& member : eg.members_of(root)) {
        size_t c = self_cost(member);
        bool ok = true;
        for (auto child : member.children) {
            const auto& sub = best(eg, child, memo);
            if (sub.first == std::numeric_limits<size_t>::max()) { ok = false; break; }
            c += sub.first;
        }
        if (ok && c < best_cost) { best_cost = c; best_node = member; found_any = true; }
    }
    (void)found_any;
    memo[root] = std::make_pair(best_cost, best_node);
    return memo[root];
}

using BuildMemo = std::unordered_map<EClassId, NodeId>;

NodeId materialize(const EGraph& eg, EClassId root, CostMemo& cmemo, BuildMemo& bmemo, Graph& out) {
    root = eg.find(root);
    auto it = bmemo.find(root);
    if (it != bmemo.end()) return it->second;


    ENode node = best(eg, root, cmemo).second;
    std::vector<NodeId> new_children;
    new_children.reserve(node.children.size());
    for (auto c : node.children) new_children.push_back(materialize(eg, c, cmemo, bmemo, out));

    NodeId new_id;
    if (node.op == OpKind::Const) {
        new_id = out.constant(node.const_value, node.shape);
    } else {
        Node n;
        n.op = node.op;
        n.inputs = new_children;
        n.shape = node.shape;
        n.const_value = node.const_value;
        n.input_slot = node.input_slot;
        n.axis = node.axis;
        n.target_shape = node.target_shape;
        new_id = out.push_raw(std::move(n));
    }
    bmemo[root] = new_id;
    return new_id;
}

}

Graph extract(const EGraph& eg, const Graph& original, const std::vector<EClassId>& node_to_eclass,
              EClassId root, NodeId& new_output) {
    Graph out;
    CostMemo cmemo;
    BuildMemo bmemo;


    for (NodeId in_id : original.inputs()) {
        EClassId cls = eg.find(node_to_eclass[in_id]);
        if (bmemo.find(cls) == bmemo.end()) {
            bmemo[cls] = out.input(original.shape_of(in_id));
        }
    }

    new_output = materialize(eg, root, cmemo, bmemo, out);
    return out;
}

}
