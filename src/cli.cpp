


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "minijax/interp.hpp"
#include "minijax/autodiff.hpp"
#include "minijax/opt.hpp"
#include "minijax/verify.hpp"
#include "minijax/fuzz.hpp"
#include "minijax/frontend.hpp"
#include "minijax/memplan.hpp"
#include "minijax/viz.hpp"
#include "minijax/nn.hpp"

using namespace minijax;

namespace {

int usage() {
    std::fprintf(stderr,
        "usage: minijax <command> [args]\n"
        "  run     <file.mjx> [--seed S]\n"
        "  grad    <file.mjx>\n"
        "  opt     <file.mjx> [--fast-math]\n"
        "  verify\n"
        "  fuzz    --iters N --seed S [--jit]\n"
        "  train   --dataset two-moons [--epochs N]\n"
        "  viz     <file.mjx> [--grad] -o out.dot\n"
        "  memplan <file.mjx>\n"
        "  repl\n");
    return 2;
}

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
    for (const auto& a : args)
        if (a == flag) return true;
    return false;
}

std::string flag_value(const std::vector<std::string>& args, const std::string& flag,
                        const std::string& fallback) {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == flag) return args[i + 1];
    return fallback;
}

frontend::LowerResult load_program(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("cannot open file: " + path);
    std::ostringstream buf;
    buf << in.rdbuf();
    return frontend::compile_source(buf.str());
}


std::vector<Tensor> make_input_values(const Graph& g, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Tensor> vals;
    for (NodeId id : g.inputs()) {
        Tensor t = Tensor::zeros(g.shape_of(id));
        for (double& v : t.data()) v = dist(rng);
        vals.push_back(std::move(t));
    }
    return vals;
}

void print_tensor(const Tensor& t) {
    std::cout << "shape=[";
    for (size_t d = 0; d < t.rank(); ++d)
        std::cout << t.shape()[d] << (d + 1 < t.rank() ? "," : "");
    std::cout << "] values=[";
    size_t shown = std::min<size_t>(t.numel(), 16);
    for (size_t i = 0; i < shown; ++i)
        std::cout << t.data()[i] << (i + 1 < shown ? ", " : "");
    if (shown < t.numel()) std::cout << ", ... (" << t.numel() << " total)";
    std::cout << "]\n";
}

std::string op_profile_table(const OpProfile& prof) {
    std::map<OpKind, double> sorted(prof.ns_per_op.begin(), prof.ns_per_op.end());
    std::ostringstream os;
    os << "per-op profile:\n  op          time(ns)   share\n";
    char line[96];
    for (const auto& [kind, ns] : sorted) {
        double share = prof.total_ns > 0 ? 100.0 * ns / prof.total_ns : 0.0;
        std::snprintf(line, sizeof(line), "  %-10s %12.0f %6.1f%%\n",
                      op_kind_name(kind), ns, share);
        os << line;
    }
    return os.str();
}

int cmd_run(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    auto prog = load_program(args[0]);
    unsigned seed = static_cast<unsigned>(std::stoul(flag_value(args, "--seed", "42")));
    auto inputs = make_input_values(prog.g, seed);
    OpProfile prof;
    auto values = eval_all(prog.g, inputs, &prof);
    std::cout << "evaluated " << prog.g.size() << " nodes; output:\n";
    print_tensor(values[prog.output]);
    std::cout << op_profile_table(prof);
    return 0;
}

int cmd_grad(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    auto prog = load_program(args[0]);
    auto inputs = make_input_values(prog.g, 42);
    auto grads = grad(prog.g, prog.output, prog.g.inputs());
    for (size_t s = 0; s < grads.size(); ++s) {
        std::cout << "d output / d input[" << s << "]: ";
        print_tensor(eval(prog.g, inputs, grads[s]));
    }
    return 0;
}

int cmd_opt(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    bool fast_math = has_flag(args, "--fast-math");
    auto prog = load_program(args[0]);
    auto inputs = make_input_values(prog.g, 42);

    auto [opt_g, opt_out] = fast_math ? optimize(prog.g, prog.output)
                                       : optimize_sound(prog.g, prog.output);
    std::cout << "nodes: " << prog.g.size() << " -> " << opt_g.size() << "\n";

    double before = eval(prog.g, inputs, prog.output).item();
    double after = eval(opt_g, inputs, opt_out).item();
    bool ok = before == after || std::fabs(before - after) < 1e-9;
    std::printf("output before=%.12g after=%.12g %s\n", before, after, ok ? "(match)" : "(MISMATCH!)");
    return ok ? 0 : 1;
}

int cmd_verify() {
    Verifier v;
    std::cout << format_soundness_report(v.soundness_report());
    return 0;
}

int cmd_fuzz(const std::vector<std::string>& args) {
    int iters = std::stoi(flag_value(args, "--iters", "50"));
    unsigned seed = static_cast<unsigned>(std::stoul(flag_value(args, "--seed", "42")));
#ifdef MINIJAX_WITH_JIT
    bool jit_oracle = has_flag(args, "--jit");
#else
    bool jit_oracle = false;
    if (has_flag(args, "--jit"))
        std::cerr << "note: built without LLVM - JIT oracle unavailable, two-oracle campaign\n";
#endif
    auto result = run_fuzz_campaign(seed, iters, 8, 4, jit_oracle);
    std::cout << "campaign: " << result.iterations_run << " programs; failures="
              << result.failures.size() << "\n";
    for (const auto& f : result.failures) {
        std::cout << "  [" << f.oracle << "] seed=" << f.seed << " depth=" << f.depth
                  << ": " << f.detail << "\n";
    }
    return result.failures.empty() ? 0 : 1;
}

int cmd_train(const std::vector<std::string>& args) {
    int epochs = std::stoi(flag_value(args, "--epochs", "60"));
    Dataset data = generate_two_moons(120, 0.08, 7);

    Graph g;
    NodeId x_node = g.input({2, 1});
    NodeId y_node = g.input({2, 1});
    MLP mlp(g, {2, 16, 16, 2}, 42);
    NodeId logits = mlp.forward(g, x_node);
    NodeId probs = g.softmax(logits);
    NodeId y_flat = g.reshape(y_node, {2});
    NodeId loss = g.cross_entropy(probs, y_flat);

    std::vector<NodeId> param_nodes;
    for (const auto& p : mlp.params) param_nodes.push_back(p.node);
    auto grad_nodes = grad(g, loss, param_nodes);

    Adam opt;
    opt.lr = 0.005;
    for (int e = 0; e < epochs; ++e) {
        double avg = train_epoch(g, mlp.params, x_node, y_node, loss, grad_nodes, data, opt);
        if (e == 0 || (e + 1) % 10 == 0 || e + 1 == epochs)
            std::printf("epoch %3d/%d  avg loss %.4f\n", e + 1, epochs, avg);
    }
    double acc = compute_accuracy(g, mlp.params, x_node, y_node, logits, data);
    std::printf("final train accuracy: %.1f%%\n", 100.0 * acc);
    return acc >= 0.95 ? 0 : 1;
}

int cmd_viz(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    std::string out_path = flag_value(args, "-o", "graph.dot");
    auto prog = load_program(args[0]);

    std::string dot;
    if (has_flag(args, "--grad")) {
        auto grads = grad(prog.g, prog.output, prog.g.inputs());
        std::vector<NodeId> hot(grads.begin(), grads.end());
        dot = to_dot(prog.g, hot);
    } else {
        dot = to_dot(prog.g);
    }
    std::ofstream of(out_path);
    if (!of) throw std::invalid_argument("cannot write file: " + out_path);
    of << dot;
    std::cout << "wrote " << out_path << " (" << prog.g.size() << " nodes)\n";
    return 0;
}

int cmd_memplan(const std::vector<std::string>& args) {
    if (args.empty()) return usage();
    auto prog = load_program(args[0]);
    MemPlan plan = plan_memory(prog.g, prog.output);
    std::cout << format_memory_report(prog.g, plan);
    return 0;
}

}


namespace {

struct ReplState {
    std::vector<std::string> history;
    Graph g;
    NodeId output = kInvalidNode;
    std::unordered_map<std::string, NodeId> bindings;
    std::vector<Tensor> input_values;
    bool has_output_stmt = false;
};


void repl_rebuild(ReplState& st, std::mt19937& rng) {
    std::ostringstream src;
    for (const auto& l : st.history) src << l << "\n";
    bool scaffold = !st.has_output_stmt;
    if (scaffold) src << "output 0.0;\n";

    frontend::LowerResult r = frontend::compile_source(src.str());
    std::vector<NodeId> user_outputs = r.outputs;
    if (scaffold && !user_outputs.empty()) user_outputs.pop_back();

    st.g = std::move(r.g);
    st.bindings = std::move(r.bindings);
    st.output = user_outputs.empty() ? kInvalidNode : user_outputs.back();


    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    while (st.input_values.size() < st.g.num_inputs()) {
        const NodeId id = st.g.inputs()[st.input_values.size()];
        Tensor t = Tensor::zeros(st.g.shape_of(id));
        for (double& v : t.data()) v = dist(rng);
        st.input_values.push_back(std::move(t));
    }
}

int cmd_repl() {
    ReplState st;
    std::mt19937 rng(42);
    std::cout << "minijax repl - enter .mjx statements one at a time; ':quit' exits\n";

    std::string line;
    while (true) {
        std::cout << "mjx> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line == ":quit" || line == ":exit") break;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

        try {
            st.history.push_back(line);
            frontend::Module m = frontend::parse(line);
            bool is_output_line = false;
            for (const auto& stmt : m.stmts)
                if (stmt.is_output) is_output_line = true;

            if (is_output_line) st.has_output_stmt = true;
            repl_rebuild(st, rng);

            if (is_output_line && st.output != kInvalidNode) {
                std::cout << "output:\n";
                print_tensor(eval(st.g, st.input_values, st.output));
            }
        } catch (const std::exception& e) {
            st.history.pop_back();
            std::cout << "error: " << e.what() << "\n";
        }
    }
    return 0;
}

}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string cmd = argv[1];
    std::vector<std::string> args(argv + 2, argv + argc);

    try {
        if (cmd == "run") return cmd_run(args);
        if (cmd == "grad") return cmd_grad(args);
        if (cmd == "opt") return cmd_opt(args);
        if (cmd == "verify") return cmd_verify();
        if (cmd == "fuzz") return cmd_fuzz(args);
        if (cmd == "train") return cmd_train(args);
        if (cmd == "viz") return cmd_viz(args);
        if (cmd == "memplan") return cmd_memplan(args);
        if (cmd == "repl") return cmd_repl();
        return usage();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
