#include "minijax/nn.hpp"
#include "minijax/interp.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace minijax {

MLP::MLP(Graph& g, const std::vector<size_t>& layer_sizes, unsigned seed) : sizes_(layer_sizes) {
    std::mt19937 rng(seed);
    for (size_t l = 0; l + 1 < layer_sizes.size(); ++l) {
        size_t in = layer_sizes[l], out = layer_sizes[l + 1];
        double limit = std::sqrt(6.0 / static_cast<double>(in + out));
        std::uniform_real_distribution<double> dist(-limit, limit);

        NodeId w_node = g.input({out, in});
        Tensor w_val = Tensor::zeros({out, in}).mapv([&](double) { return dist(rng); });
        params.push_back({w_node, w_val, "W" + std::to_string(l)});

        NodeId b_node = g.input({out, 1});
        Tensor b_val = Tensor::zeros({out, 1});
        params.push_back({b_node, b_val, "b" + std::to_string(l)});
    }
}

NodeId MLP::forward(Graph& g, NodeId x) const {
    NodeId cur = x;
    size_t num_layers = sizes_.size() - 1;
    for (size_t l = 0; l < num_layers; ++l) {
        NodeId w_node = params[l * 2].node;
        NodeId b_node = params[l * 2 + 1].node;
        NodeId pre = g.add(g.matmul(w_node, cur), b_node);
        cur = (l + 1 < num_layers) ? g.relu(pre) : pre;
    }
    return g.reshape(cur, {sizes_.back()});
}

void SGD::step(std::vector<Param>& params, const std::vector<Tensor>& grads) {
    for (size_t i = 0; i < params.size(); ++i) {
        params[i].value = params[i].value - grads[i].mapv([&](double x) { return x * lr; });
    }
}

void Adam::step(std::vector<Param>& params, const std::vector<Tensor>& grads) {
    if (m.empty()) {
        m.reserve(params.size());
        v.reserve(params.size());
        for (const auto& p : params) {
            m.push_back(Tensor::zeros(p.value.shape()));
            v.push_back(Tensor::zeros(p.value.shape()));
        }
    }
    ++t;
    double bias_correction1 = 1.0 - std::pow(beta1, t);
    double bias_correction2 = 1.0 - std::pow(beta2, t);

    for (size_t i = 0; i < params.size(); ++i) {
        m[i] = Tensor::zip_map(m[i], grads[i], [&](double mi, double gi) { return beta1 * mi + (1 - beta1) * gi; });
        v[i] = Tensor::zip_map(v[i], grads[i],
                                [&](double vi, double gi) { return beta2 * vi + (1 - beta2) * gi * gi; });
        Tensor m_hat = m[i].mapv([&](double x) { return x / bias_correction1; });
        Tensor v_hat = v[i].mapv([&](double x) { return x / bias_correction2; });
        Tensor update = Tensor::zip_map(m_hat, v_hat, [&](double mh, double vh) { return lr * mh / (std::sqrt(vh) + eps); });
        params[i].value = params[i].value - update;
    }
}

Dataset generate_two_moons(size_t n_per_class, double noise, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise_dist(0.0, noise);
    Dataset ds;
    for (size_t i = 0; i < n_per_class; ++i) {
        double theta = M_PI * static_cast<double>(i) / static_cast<double>(n_per_class - 1);

        double x0 = std::cos(theta) + noise_dist(rng);
        double y0 = std::sin(theta) + noise_dist(rng);
        ds.xs.push_back(Tensor::from_vec({2, 1}, {x0, y0}));
        ds.ys.push_back(Tensor::from_vec({2, 1}, {1.0, 0.0}));

        double x1 = 1.0 - std::cos(theta) + noise_dist(rng);
        double y1 = 1.0 - std::sin(theta) - 0.5 + noise_dist(rng);
        ds.xs.push_back(Tensor::from_vec({2, 1}, {x1, y1}));
        ds.ys.push_back(Tensor::from_vec({2, 1}, {0.0, 1.0}));
    }
    return ds;
}

std::vector<Tensor> eval_with_params(const Graph& g, const std::vector<Param>& params, NodeId x_node,
                                      const Tensor& x_val, NodeId y_node, const Tensor& y_val) {
    std::unordered_map<NodeId, const Tensor*> lookup;
    for (const auto& p : params) lookup[p.node] = &p.value;
    lookup[x_node] = &x_val;
    lookup[y_node] = &y_val;

    std::vector<Tensor> inputs;
    inputs.reserve(g.num_inputs());
    for (NodeId in_id : g.inputs()) {
        auto it = lookup.find(in_id);
        if (it == lookup.end()) throw std::runtime_error("eval_with_params: missing value for an input node");
        inputs.push_back(*it->second);
    }
    return eval_all(g, inputs);
}

double train_epoch(Graph& g, std::vector<Param>& params, NodeId x_node, NodeId y_node, NodeId loss_node,
                    const std::vector<NodeId>& grad_nodes, const Dataset& data, Adam& opt) {
    double total_loss = 0.0;
    for (size_t i = 0; i < data.xs.size(); ++i) {
        auto values = eval_with_params(g, params, x_node, data.xs[i], y_node, data.ys[i]);
        total_loss += values[loss_node].item();

        std::vector<Tensor> grads;
        grads.reserve(grad_nodes.size());
        for (NodeId gid : grad_nodes) grads.push_back(values[gid]);
        opt.step(params, grads);
    }
    return total_loss / static_cast<double>(data.xs.size());
}

double compute_accuracy(const Graph& g, const std::vector<Param>& params, NodeId x_node, NodeId y_node,
                         NodeId logits_node, const Dataset& data) {
    size_t correct = 0;
    for (size_t i = 0; i < data.xs.size(); ++i) {
        auto values = eval_with_params(g, params, x_node, data.xs[i], y_node, data.ys[i]);
        const Tensor& logits = values[logits_node];
        size_t pred = std::distance(logits.data().begin(),
                                     std::max_element(logits.data().begin(), logits.data().end()));
        size_t label = std::distance(data.ys[i].data().begin(),
                                      std::max_element(data.ys[i].data().begin(), data.ys[i].data().end()));
        if (pred == label) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(data.xs.size());
}

}
