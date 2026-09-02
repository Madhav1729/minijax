#include "minijax/memplan.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <sstream>

namespace minijax {


std::vector<size_t> compute_last_use(size_t num_values, const std::vector<StepAccess>& steps) {
    std::vector<size_t> last_use(num_values, 0);
    for (size_t i = 0; i < steps.size(); ++i) {
        for (uint32_t r : steps[i].reads) {
            last_use[r] = std::max(last_use[r], i);
        }
    }
    return last_use;
}

namespace {

bool is_elementwise(OpKind k) {
    switch (k) {
        case OpKind::Add: case OpKind::Sub: case OpKind::Mul: case OpKind::Div:
        case OpKind::Neg: case OpKind::Relu: case OpKind::Step: case OpKind::Tanh:
        case OpKind::Sigmoid: case OpKind::Exp: case OpKind::Log: case OpKind::Sqrt:
        case OpKind::Abs:
            return true;
        default:
            return false;
    }
}

size_t numel_of(const std::vector<size_t>& shape) {
    return shape.empty()
               ? 1
               : std::accumulate(shape.begin(), shape.end(), size_t{1},
                                  [](size_t a, size_t b) { return a * b; });
}

}

MemPlan plan_memory(const Graph& g, NodeId output) {
    if (output >= g.size()) {
        throw std::invalid_argument("plan_memory: output node id out of range");
    }


    std::vector<bool> reachable(g.size(), false);
    {
        std::vector<NodeId> stack = {output};
        while (!stack.empty()) {
            NodeId id = stack.back();
            stack.pop_back();
            if (reachable[id]) continue;
            reachable[id] = true;
            for (NodeId in : g.node(id).inputs) stack.push_back(in);
        }
    }


    std::vector<size_t> consumers(g.size(), 0);
    for (NodeId i = 0; i < g.size(); ++i) {
        if (!reachable[i]) continue;
        for (NodeId in : g.node(i).inputs) ++consumers[in];
    }

    MemPlan plan;
    plan.slot_of_node.assign(g.size(), SIZE_MAX);


    std::vector<StepAccess> steps;
    steps.reserve(g.size());
    for (NodeId i = 0; i < g.size(); ++i) {
        if (!reachable[i]) continue;
        StepAccess sa;
        sa.dst = static_cast<uint32_t>(i);
        for (NodeId in : g.node(i).inputs) sa.reads.push_back(static_cast<uint32_t>(in));
        steps.push_back(std::move(sa));
    }
    std::vector<size_t> last_use = compute_last_use(g.size(), steps);
    last_use[output] = SIZE_MAX;


    std::vector<size_t> die_step(g.size(), SIZE_MAX);
    for (NodeId v = 0; v < g.size(); ++v) {
        if (!reachable[v] || last_use[v] == SIZE_MAX) continue;
        die_step[v] = (consumers[v] == 0) ? v + 1 : last_use[v] + 1;
    }


    struct Slot {
        size_t capacity_numel;
        bool live;
    };
    std::vector<Slot> slots;

    std::map<size_t, std::vector<size_t>> free_buckets;

    size_t live_bytes = 0;

    auto release_slot_of = [&](NodeId v) {
        size_t s = plan.slot_of_node[v];
        slots[s].live = false;
        live_bytes -= slots[s].capacity_numel * sizeof(double);
        free_buckets[slots[s].capacity_numel].push_back(s);
    };

    for (NodeId v = 0; v < g.size(); ++v) {
        if (!reachable[v]) continue;


        for (NodeId u = 0; u < g.size(); ++u) {
            if (!reachable[u] || u == v) continue;
            if (plan.slot_of_node[u] == SIZE_MAX) continue;
            size_t s = plan.slot_of_node[u];
            if (slots[s].live && die_step[u] != SIZE_MAX && die_step[u] <= v) {
                release_slot_of(u);
            }
        }

        size_t n = numel_of(g.node(v).shape);
        size_t slot;
        auto bucket = free_buckets.find(n);
        if (bucket != free_buckets.end() && !bucket->second.empty()) {
            slot = bucket->second.back();
            bucket->second.pop_back();
            slots[slot].live = true;
            live_bytes += n * sizeof(double);
        } else {
            slot = slots.size();
            slots.push_back({n, true});
            live_bytes += n * sizeof(double);
        }
        plan.slot_of_node[v] = slot;

        plan.peak_bytes_planned = std::max(plan.peak_bytes_planned, live_bytes);
    }
    plan.num_slots = slots.size();


    for (NodeId v = 0; v < g.size(); ++v) {
        if (reachable[v]) plan.peak_bytes_unplanned += numel_of(g.node(v).shape) * sizeof(double);
    }


    for (NodeId u = 0; u < g.size(); ++u) {
        if (!reachable[u] || !is_elementwise(g.node(u).op)) continue;
        for (NodeId p : g.node(u).inputs) {
            if (p == u || !reachable[p]) continue;
            const Node& pn = g.node(p);
            if (pn.op == OpKind::Input || p == output) continue;
            if (consumers[p] != 1) continue;
            if (g.shape_of(p) != g.shape_of(u)) continue;
            plan.in_place_candidates.push_back({p, u});
        }
    }


    auto links_to = [&](NodeId p, NodeId c) {
        return is_elementwise(g.node(p).op) && is_elementwise(g.node(c).op) &&
               consumers[p] == 1 && g.shape_of(p) == g.shape_of(c);
    };

    std::vector<NodeId> sole_consumer(g.size(), kInvalidNode);
    for (NodeId i = 0; i < g.size(); ++i) {
        if (!reachable[i]) continue;
        for (NodeId in : g.node(i).inputs) {
            if (sole_consumer[in] == kInvalidNode) sole_consumer[in] = i;
            else sole_consumer[in] = kInvalidNode - 1;
        }
    }
    auto has_sole_consumer = [&](NodeId v, NodeId& out_c) {
        NodeId s = sole_consumer[v];
        if (s == kInvalidNode || s == kInvalidNode - 1) return false;
        out_c = s;
        return true;
    };
    for (NodeId v = 0; v < g.size(); ++v) {
        if (!reachable[v] || !is_elementwise(g.node(v).op)) continue;

        bool starts = true;
        for (NodeId p : g.node(v).inputs) {
            if (links_to(p, v)) { starts = false; break; }
        }
        if (!starts) continue;
        std::vector<NodeId> chain = {v};
        NodeId cur = v;
        NodeId next;
        while (has_sole_consumer(cur, next) && links_to(cur, next)) {
            chain.push_back(next);
            cur = next;
        }
        if (chain.size() >= 2) plan.fusion_chains.push_back(std::move(chain));
    }

    return plan;
}

std::string format_memory_report(const Graph& g, const MemPlan& plan) {
    std::ostringstream os;
    os << "node  op          shape       slot\n";
    os << "----------------------------------\n";
    for (NodeId i = 0; i < g.size(); ++i) {
        if (plan.slot_of_node[i] == SIZE_MAX) continue;
        const Node& n = g.node(i);
        os << i << "     " << op_kind_name(n.op);
        os << "     [";
        for (size_t d = 0; d < n.shape.size(); ++d) {
            if (d) os << ",";
            os << n.shape[d];
        }
        os << "]     " << plan.slot_of_node[i] << "\n";
    }
    os << "\nin-place candidates:\n";
    if (plan.in_place_candidates.empty()) os << "  (none)\n";
    for (const auto& ip : plan.in_place_candidates) {
        os << "  node " << ip.consumer << " may write in place into node " << ip.producer << "\n";
    }
    os << "\nfusion chains:\n";
    if (plan.fusion_chains.empty()) os << "  (none)\n";
    for (const auto& ch : plan.fusion_chains) {
        os << "  ";
        for (NodeId v : ch) os << v << " -> ";
        os << "(end)\n";
    }
    os << "\npeak memory (unplanned): " << plan.peak_bytes_unplanned << " bytes\n"
       << "peak memory (planned):   " << plan.peak_bytes_planned << " bytes\n"
       << "reduction: " << static_cast<int>(plan.reduction_ratio() * 100.0) << "%\n";
    return os.str();
}

}
