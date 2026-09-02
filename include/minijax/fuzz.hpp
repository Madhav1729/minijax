#pragma once


#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include "minijax/ir.hpp"
#include "minijax/tensor.hpp"

#if defined(MINIJAX_WITH_JIT)
#define MINIJAX_FUZZ_HAS_JIT_ORACLE 1
#endif

namespace minijax {


struct GeneratedProgram {
    Graph g;
    NodeId output;
    std::vector<Tensor> inputs;
};

GeneratedProgram generate_random_program(unsigned seed, int max_depth = 8, int max_dim = 4, int num_inputs = 3);

struct OracleResult {
    bool passed;
    std::string detail;
};


OracleResult check_interp_vs_vm(const Graph& g, NodeId output, const std::vector<Tensor>& inputs,
                                 double tol = 1e-9);


OracleResult check_metamorphic_optimize_sound(const Graph& g, NodeId output, const std::vector<Tensor>& inputs,
                                               double tol = 1e-6);

#if defined(MINIJAX_FUZZ_HAS_JIT_ORACLE)


OracleResult check_interp_vs_jit(const Graph& g, NodeId output, const std::vector<Tensor>& inputs,
                                  double tol = 1e-9);
#endif

struct FuzzFailure {
    unsigned seed;
    int depth;
    std::string oracle;
    std::string detail;
    std::string signature;
};

struct CampaignResult {
    int iterations_run = 0;
    std::vector<FuzzFailure> failures;
};


CampaignResult run_fuzz_campaign(unsigned base_seed, int iterations, int max_depth = 8, int max_dim = 4,
                                  bool with_jit_oracle = false);

}
