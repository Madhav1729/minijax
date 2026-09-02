#pragma once


#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "minijax/ir.hpp"

namespace minijax {


struct StepAccess {
    uint32_t dst;
    std::vector<uint32_t> reads;
};


std::vector<size_t> compute_last_use(size_t num_values, const std::vector<StepAccess>& steps);


struct MemPlan {


    std::vector<size_t> slot_of_node;
    size_t num_slots = 0;


    size_t peak_bytes_unplanned = 0;
    size_t peak_bytes_planned = 0;

    struct InPlaceOp {
        NodeId producer;
        NodeId consumer;
    };


    std::vector<InPlaceOp> in_place_candidates;


    std::vector<std::vector<NodeId>> fusion_chains;

    double reduction_ratio() const {
        return peak_bytes_unplanned == 0
                   ? 0.0
                   : 1.0 - static_cast<double>(peak_bytes_planned) /
                               static_cast<double>(peak_bytes_unplanned);
    }
};

MemPlan plan_memory(const Graph& g, NodeId output);


std::string format_memory_report(const Graph& g, const MemPlan& plan);

}
