#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "ir.hpp"

namespace minijax {

// A tiny optimizer pass: constant folding for binary ops when both inputs are Const.
inline void const_fold(Graph& g) {
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        Node& node = g.nodes[i];
        // binary ops
        if (node.inputs.size() >= 2) {
            size_t a_id = node.inputs[0];
            size_t b_id = node.inputs[1];
            const Node& a = g.nodes[a_id];
            const Node& b = g.nodes[b_id];

            // both const -> fold
            if (a.op == OpKind::Const && b.op == OpKind::Const) {
                double av = a.const_value;
                double bv = b.const_value;
                double rv = 0.0;
                switch (node.op) {
                    case OpKind::Add: rv = av + bv; break;
                    case OpKind::Sub: rv = av - bv; break;
                    case OpKind::Mul: rv = av * bv; break;
                    case OpKind::Div: rv = av / bv; break;
                    default: break;
                }
                node.op = OpKind::Const;
                node.inputs.clear();
                node.const_value = rv;
                continue;
            }

            // identity removals: x + 0 -> x, x - 0 -> x, x * 1 -> x, x / 1 -> x
            if (node.op == OpKind::Add) {
                if (a.op == OpKind::Const && a.const_value == 0.0) {
                    node = g.nodes[b_id];
                    continue;
                }
                if (b.op == OpKind::Const && b.const_value == 0.0) {
                    node = g.nodes[a_id];
                    continue;
                }
            } else if (node.op == OpKind::Sub) {
                if (b.op == OpKind::Const && b.const_value == 0.0) {
                    node = g.nodes[a_id];
                    continue;
                }
            } else if (node.op == OpKind::Mul) {
                if (a.op == OpKind::Const) {
                    if (a.const_value == 0.0) {
                        node.op = OpKind::Const; node.inputs.clear(); node.const_value = 0.0; continue;
                    }
                    if (a.const_value == 1.0) { node = g.nodes[b_id]; continue; }
                }
                if (b.op == OpKind::Const) {
                    if (b.const_value == 0.0) {
                        node.op = OpKind::Const; node.inputs.clear(); node.const_value = 0.0; continue;
                    }
                    if (b.const_value == 1.0) { node = g.nodes[a_id]; continue; }
                }
            } else if (node.op == OpKind::Div) {
                if (b.op == OpKind::Const && b.const_value == 1.0) {
                    node = g.nodes[a_id];
                    continue;
                }
            }
        }
    }
}

inline void commutative_rewrite(Graph& g) {
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        Node& node = g.nodes[i];
        if (!(node.op == OpKind::Add || node.op == OpKind::Mul)) continue;
        if (node.inputs.size() < 2) continue;
        size_t l = node.inputs[0];
        size_t r = node.inputs[1];
        const Node& ln = g.nodes[l];
        const Node& rn = g.nodes[r];

        // If left is const and right not, swap to canonicalize const on the right
        if (ln.op == OpKind::Const && rn.op != OpKind::Const) {
            std::swap(node.inputs[0], node.inputs[1]);
            std::swap(l, r);
            // update views
        }

        // Pattern: (Const + (X + Const2)) -> ((X) + ConstSum)
        if (node.op == OpKind::Add) {
            if (g.nodes[r].op == OpKind::Add) {
                size_t r0 = g.nodes[r].inputs[0];
                size_t r1 = g.nodes[r].inputs[1];
                // if left is Const and one child of right is Const, combine
                if (g.nodes[l].op == OpKind::Const) {
                    if (g.nodes[r0].op == OpKind::Const) {
                        double s = g.nodes[l].const_value + g.nodes[r0].const_value;
                        // place combined const into r0
                        g.nodes[r0].const_value = s;
                        // replace current node with r (copy)
                        node = g.nodes[r];
                        continue;
                    }
                    if (g.nodes[r1].op == OpKind::Const) {
                        double s = g.nodes[l].const_value + g.nodes[r1].const_value;
                        g.nodes[r1].const_value = s;
                        node = g.nodes[r];
                        continue;
                    }
                }
            }
        }

        // Pattern: (Const * (X * Const2)) -> ((X) * ConstProd)
        if (node.op == OpKind::Mul) {
            if (g.nodes[r].op == OpKind::Mul) {
                size_t r0 = g.nodes[r].inputs[0];
                size_t r1 = g.nodes[r].inputs[1];
                if (g.nodes[l].op == OpKind::Const) {
                    if (g.nodes[r0].op == OpKind::Const) {
                        double p = g.nodes[l].const_value * g.nodes[r0].const_value;
                        g.nodes[r0].const_value = p;
                        node = g.nodes[r];
                        continue;
                    }
                    if (g.nodes[r1].op == OpKind::Const) {
                        double p = g.nodes[l].const_value * g.nodes[r1].const_value;
                        g.nodes[r1].const_value = p;
                        node = g.nodes[r];
                        continue;
                    }
                }
            }
        }
    }
}

inline void optimize(Graph& g) {
    // currently a single-pass const folding; later add more passes
    // order of passes: commutative rewrite then const-fold
    commutative_rewrite(g);
    const_fold(g);
}

// Top-level optimize sequence with rebuild
// optimize_all is defined after helper passes

// Rebuild a new graph with associative flattening for Add and Mul.
inline void rebuild_with_assoc(Graph& g) {
    Graph g2;
    size_t n = g.nodes.size();
    std::vector<size_t> map(n, 0);

    // helper to flatten original graph for a target op
    std::function<void(size_t, OpKind, std::vector<size_t>&)> collect_leaves;
    collect_leaves = [&](size_t id, OpKind target, std::vector<size_t>& out) {
        const Node& node = g.nodes[id];
        if (node.op == target) {
            for (size_t child : node.inputs) collect_leaves(child, target, out);
        } else {
            out.push_back(id);
        }
    };

    // keep a copy of original nodes for slot/shape info
    std::vector<Node> old_nodes = g.nodes;

    for (size_t i = 0; i < n; ++i) {
        const Node& node = old_nodes[i];
        switch (node.op) {
            case OpKind::Input: {
                size_t nid = g2.input(node.shape);
                map[i] = nid;
                break;
            }
            case OpKind::Const: {
                size_t nid = g2.constant(node.const_value, node.shape);
                map[i] = nid;
                break;
            }
            case OpKind::Add:
            case OpKind::Mul: {
                OpKind target = node.op;
                std::vector<size_t> leaves;
                collect_leaves(i, target, leaves);

                // separate consts and non-consts, with canonical ordering for non-consts
                double const_acc = (target == OpKind::Add) ? 0.0 : 1.0;
                std::vector<std::pair<std::string, size_t>> nonconst_pairs; // (key, new_id)

                // structural fingerprint helper for deterministic ordering
                std::unordered_map<size_t, std::string> fp_memo;
                std::function<std::string(size_t)> fingerprint = [&](size_t id)->std::string {
                    if (fp_memo.count(id)) return fp_memo[id];
                    const Node& nd = g.nodes[id];
                    std::string s = std::to_string(static_cast<int>(nd.op));
                    s += ":";
                    for (size_t sh : nd.shape) { s += std::to_string(sh) + ","; }
                    s += "(";
                    for (size_t inp : nd.inputs) {
                        s += fingerprint(inp);
                        s += ";";
                    }
                    s += ")";
                    fp_memo[id] = s;
                    return s;
                };

                for (size_t leaf_old : leaves) {
                    const Node& ln = g.nodes[leaf_old];
                    if (ln.op == OpKind::Const) {
                        if (target == OpKind::Add) const_acc += ln.const_value;
                        else const_acc *= ln.const_value;
                    } else {
                        // include original node id as deterministic tie-breaker
                        std::string key = fingerprint(leaf_old) + "_" + std::to_string(leaf_old);
                        nonconst_pairs.emplace_back(key, map[leaf_old]);
                    }
                }

                // canonical order: sort by key (stable)
                std::stable_sort(nonconst_pairs.begin(), nonconst_pairs.end(), [](auto &a, auto &b){ return a.first < b.first; });

                // build combined node in g2 from ordered non-const operands
                size_t cur_id = 0;
                if (nonconst_pairs.empty()) {
                    // pure const
                    cur_id = g2.constant(const_acc, node.shape);
                } else {
                    cur_id = nonconst_pairs[0].second;
                    for (size_t k = 1; k < nonconst_pairs.size(); ++k) {
                        if (target == OpKind::Add) cur_id = g2.add(cur_id, nonconst_pairs[k].second);
                        else cur_id = g2.mul(cur_id, nonconst_pairs[k].second);
                    }
                    bool need_const = (target == OpKind::Add) ? (const_acc != 0.0) : (const_acc != 1.0);
                    if (need_const) {
                        size_t cid = g2.constant(const_acc, node.shape);
                        if (target == OpKind::Add) cur_id = g2.add(cur_id, cid);
                        else cur_id = g2.mul(cur_id, cid);
                    }
                }
                map[i] = cur_id;
                break;
            }
            default: {
                // generic: recreate node with mapped inputs
                std::vector<size_t> new_inputs;
                new_inputs.reserve(node.inputs.size());
                for (size_t in : node.inputs) new_inputs.push_back(map[in]);
                size_t nid = g2.push(node.op, new_inputs, node.shape, node.const_value, node.axis);
                map[i] = nid;
                break;
            }
        }
    }

    // replace original graph with rebuilt one
    g.nodes = std::move(g2.nodes);
    g.inputs = std::move(g2.inputs);
}

inline std::string graph_fingerprint(const Graph& g) {
    std::string out;
    out.reserve(g.nodes.size() * 16);
    for (const auto& n : g.nodes) {
        out.push_back(static_cast<char>(n.op));
        out.push_back('|');
        for (size_t i = 0; i < n.inputs.size(); ++i) {
            out += std::to_string(n.inputs[i]);
            out.push_back(',');
        }
        out.push_back('|');
        if (n.op == OpKind::Const) {
            out += std::to_string(n.const_value);
            out.push_back('|');
        }
        for (size_t s : n.shape) {
            out += std::to_string(s);
            out.push_back(',');
        }
        out.push_back('\n');
    }
    return out;
}

inline void optimize_until_fixedpoint(Graph& g, size_t max_iters = 16) {
    std::string prev = graph_fingerprint(g);
    for (size_t it = 0; it < max_iters; ++it) {
        optimize(g);
        std::string cur = graph_fingerprint(g);
        if (cur == prev) return;
        prev = std::move(cur);
    }
}

// Simple cost model: subtree size and extractor that rebuilds Add/Mul by ordering operands
inline size_t compute_subtree_sizes(const Graph& g, std::vector<size_t>& sizes) {
    sizes.assign(g.nodes.size(), 0);
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        sizes[i] = 1; // count self
    }
    // nodes are topologically ordered (assumed), accumulate children sizes into parents
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        for (size_t in : g.nodes[i].inputs) {
            sizes[i] += sizes[in];
        }
    }
    return sizes.size();
}

inline void extract_min_cost(Graph& g) {
    // Build a new graph that reorders associative commutative operands by subtree size
    Graph g2;
    size_t n = g.nodes.size();
    std::vector<size_t> map(n, 0);
    std::vector<size_t> sizes;
    compute_subtree_sizes(g, sizes);

    // helper to collect leaves like rebuild_with_assoc
    std::function<void(size_t, OpKind, std::vector<size_t>&)> collect_leaves;
    collect_leaves = [&](size_t id, OpKind target, std::vector<size_t>& out) {
        const Node& node = g.nodes[id];
        if (node.op == target) {
            for (size_t child : node.inputs) collect_leaves(child, target, out);
        } else {
            out.push_back(id);
        }
    };

    for (size_t i = 0; i < n; ++i) {
        const Node& node = g.nodes[i];
        switch (node.op) {
            case OpKind::Input: {
                size_t nid = g2.input(node.shape);
                map[i] = nid;
                break;
            }
            case OpKind::Const: {
                size_t nid = g2.constant(node.const_value, node.shape);
                map[i] = nid;
                break;
            }
            case OpKind::Add:
            case OpKind::Mul: {
                OpKind target = node.op;
                std::vector<size_t> leaves;
                collect_leaves(i, target, leaves);

                // order leaves by increasing subtree size to reduce peak cost
                std::stable_sort(leaves.begin(), leaves.end(), [&](size_t a, size_t b){
                    return sizes[a] < sizes[b];
                });

                // build chain
                size_t cur = 0;
                if (leaves.empty()) {
                    cur = (target == OpKind::Add) ? g2.constant(0.0, node.shape) : g2.constant(1.0, node.shape);
                } else {
                    cur = map[leaves[0]];
                    for (size_t k = 1; k < leaves.size(); ++k) {
                        size_t nid = map[leaves[k]];
                        if (target == OpKind::Add) cur = g2.add(cur, nid);
                        else cur = g2.mul(cur, nid);
                    }
                }
                map[i] = cur;
                break;
            }
            default: {
                std::vector<size_t> new_inputs;
                new_inputs.reserve(node.inputs.size());
                for (size_t in : node.inputs) new_inputs.push_back(map[in]);
                size_t nid = g2.push(node.op, new_inputs, node.shape, node.const_value, node.axis);
                map[i] = nid;
                break;
            }
        }
    }

    // replace graph
    g.nodes = std::move(g2.nodes);
    g.inputs = std::move(g2.inputs);
}

// FLOPs-based cost model: estimate work per node
inline void compute_flop_costs(const Graph& g, std::vector<double>& cost) {
    cost.assign(g.nodes.size(), 0.0);
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& n = g.nodes[i];
        double elems = 1.0;
        for (size_t s : n.shape) elems *= static_cast<double>(s == 0 ? 1 : s);
        switch (n.op) {
            case OpKind::Add:
            case OpKind::Sub:
            case OpKind::Mul:
            case OpKind::Div:
                cost[i] = elems; // elementwise ops cost ~ #elements
                break;
            case OpKind::MatMul: {
                // assume shape [m,k] * [k,n] -> cost ~ 2*m*n*k
                if (n.inputs.size() >= 2) {
                    const auto& A = g.nodes[n.inputs[0]].shape;
                    const auto& B = g.nodes[n.inputs[1]].shape;
                    if (A.size() == 2 && B.size() == 2) {
                        double m = static_cast<double>(A[0]);
                        double k = static_cast<double>(A[1]);
                        double ncol = static_cast<double>(B[1]);
                        cost[i] = 2.0 * m * k * ncol;
                    }
                }
                break;
            }
            case OpKind::Broadcast:
            case OpKind::Reshape:
                cost[i] = 0.1 * elems; // cheap
                break;
            case OpKind::Sum:
            case OpKind::SumAxis:
                cost[i] = elems; // reduction cost proportional
                break;
            default:
                cost[i] = elems;
                break;
        }
    }
    // accumulate upstream: add children's cost to parent
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        for (size_t in : g.nodes[i].inputs) cost[i] += cost[in];
    }
}

// Extractor that orders associative operands by FLOPs cost (small-first)
inline void extract_by_flops(Graph& g) {
    Graph g2;
    size_t n = g.nodes.size();
    std::vector<size_t> map(n, 0);
    std::vector<double> costs;
    compute_flop_costs(g, costs);

    std::function<void(size_t, OpKind, std::vector<size_t>&)> collect_leaves;
    collect_leaves = [&](size_t id, OpKind target, std::vector<size_t>& out) {
        const Node& node = g.nodes[id];
        if (node.op == target) {
            for (size_t child : node.inputs) collect_leaves(child, target, out);
        } else {
            out.push_back(id);
        }
    };

    for (size_t i = 0; i < n; ++i) {
        const Node& node = g.nodes[i];
        switch (node.op) {
            case OpKind::Input: map[i] = g2.input(node.shape); break;
            case OpKind::Const: map[i] = g2.constant(node.const_value, node.shape); break;
            case OpKind::Add:
            case OpKind::Mul: {
                OpKind target = node.op;
                std::vector<size_t> leaves;
                collect_leaves(i, target, leaves);
                std::stable_sort(leaves.begin(), leaves.end(), [&](size_t a, size_t b){ return costs[a] < costs[b]; });
                size_t cur = (leaves.empty() ? (target==OpKind::Add ? g2.constant(0.0, node.shape) : g2.constant(1.0, node.shape)) : map[leaves[0]]);
                for (size_t k = 1; k < leaves.size(); ++k) {
                    size_t nid = map[leaves[k]];
                    cur = (target==OpKind::Add) ? g2.add(cur, nid) : g2.mul(cur, nid);
                }
                map[i] = cur;
                break;
            }
            default: {
                std::vector<size_t> new_inputs;
                for (size_t in : node.inputs) new_inputs.push_back(map[in]);
                map[i] = g2.push(node.op, new_inputs, node.shape, node.const_value, node.axis);
                break;
            }
        }
    }

    g.nodes = std::move(g2.nodes);
    g.inputs = std::move(g2.inputs);
}

// Combined FLOPs + memory cost model
inline void compute_flop_mem_costs(const Graph& g, std::vector<double>& cost, double flop_weight = 1.0, double mem_weight = 0.1) {
    std::vector<double> flops(g.nodes.size());
    // reuse compute_flop_costs logic for flops
    for (size_t i = 0; i < g.nodes.size(); ++i) flops[i] = 0.0;
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        const Node& n = g.nodes[i];
        double elems = 1.0;
        for (size_t s : n.shape) elems *= static_cast<double>(s == 0 ? 1 : s);
        switch (n.op) {
            case OpKind::Add:
            case OpKind::Sub:
            case OpKind::Mul:
            case OpKind::Div:
                flops[i] = elems;
                break;
            case OpKind::MatMul: {
                if (n.inputs.size() >= 2) {
                    const auto& A = g.nodes[n.inputs[0]].shape;
                    const auto& B = g.nodes[n.inputs[1]].shape;
                    if (A.size() == 2 && B.size() == 2) {
                        double m = static_cast<double>(A[0]);
                        double k = static_cast<double>(A[1]);
                        double ncol = static_cast<double>(B[1]);
                        flops[i] = 2.0 * m * k * ncol;
                    }
                }
                break;
            }
            default:
                flops[i] = elems;
                break;
        }
    }
    // accumulate
    for (size_t i = 0; i < g.nodes.size(); ++i) for (size_t in : g.nodes[i].inputs) flops[i] += flops[in];

    // memory estimate: bytes touched ~ #elements * sizeof(double)
    std::vector<double> mem(g.nodes.size(), 0.0);
    for (size_t i = 0; i < g.nodes.size(); ++i) {
        double elems = 1.0;
        for (size_t s : g.nodes[i].shape) elems *= static_cast<double>(s == 0 ? 1 : s);
        mem[i] = elems * 8.0; // bytes
    }
    for (size_t i = 0; i < g.nodes.size(); ++i) for (size_t in : g.nodes[i].inputs) mem[i] += mem[in];

    cost.assign(g.nodes.size(), 0.0);
    for (size_t i = 0; i < g.nodes.size(); ++i) cost[i] = flop_weight * flops[i] + mem_weight * mem[i];
}

// Extractor using combined flop+mem costs
inline void extract_by_cost(Graph& g, double flop_weight = 1.0, double mem_weight = 0.05) {
    Graph g2;
    size_t n = g.nodes.size();
    std::vector<size_t> map(n, 0);
    std::vector<double> costs;
    compute_flop_mem_costs(g, costs, flop_weight, mem_weight);

    std::function<void(size_t, OpKind, std::vector<size_t>&)> collect_leaves;
    collect_leaves = [&](size_t id, OpKind target, std::vector<size_t>& out) {
        const Node& node = g.nodes[id];
        if (node.op == target) {
            for (size_t child : node.inputs) collect_leaves(child, target, out);
        } else {
            out.push_back(id);
        }
    };

    for (size_t i = 0; i < n; ++i) {
        const Node& node = g.nodes[i];
        switch (node.op) {
            case OpKind::Input: { map[i] = g2.input(node.shape); break; }
            case OpKind::Const: { map[i] = g2.constant(node.const_value, node.shape); break; }
            case OpKind::Add:
            case OpKind::Mul: {
                OpKind target = node.op;
                std::vector<size_t> leaves;
                collect_leaves(i, target, leaves);
                std::stable_sort(leaves.begin(), leaves.end(), [&](size_t a, size_t b){ return costs[a] < costs[b]; });
                size_t cur = (leaves.empty() ? (target==OpKind::Add ? g2.constant(0.0, node.shape) : g2.constant(1.0, node.shape)) : map[leaves[0]]);
                for (size_t k = 1; k < leaves.size(); ++k) {
                    size_t nid = map[leaves[k]];
                    cur = (target==OpKind::Add) ? g2.add(cur, nid) : g2.mul(cur, nid);
                }
                map[i] = cur;
                break;
            }
            default: {
                std::vector<size_t> new_inputs;
                for (size_t in : node.inputs) new_inputs.push_back(map[in]);
                map[i] = g2.push(node.op, new_inputs, node.shape, node.const_value, node.axis);
                break;
            }
        }
    }

    g.nodes = std::move(g2.nodes);
    g.inputs = std::move(g2.inputs);
}

// Top-level optimize sequence with rebuild
inline void optimize_all(Graph& g) {
    // rebuild graph to flatten assoc ops, then run fixed-point optimize
    rebuild_with_assoc(g);
    optimize_until_fixedpoint(g);
}

} // namespace minijax
