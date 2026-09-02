#pragma once


#include <vector>
#include <string>
#include <random>
#include "minijax/ir.hpp"
#include "minijax/tensor.hpp"

namespace minijax {

struct Param {
    NodeId node;
    Tensor value;
    std::string name;
};


class MLP {
public:


    MLP(Graph& g, const std::vector<size_t>& layer_sizes, unsigned seed);


    NodeId forward(Graph& g, NodeId x) const;

    std::vector<Param> params;

private:
    std::vector<size_t> sizes_;
};


struct SGD {
    double lr = 0.1;
    void step(std::vector<Param>& params, const std::vector<Tensor>& grads);
};

struct Adam {
    double lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    std::vector<Tensor> m, v;
    int t = 0;
    void step(std::vector<Param>& params, const std::vector<Tensor>& grads);
};


struct Dataset {
    std::vector<Tensor> xs;
    std::vector<Tensor> ys;
};


Dataset generate_two_moons(size_t n_per_class, double noise, unsigned seed);


std::vector<Tensor> eval_with_params(const Graph& g, const std::vector<Param>& params, NodeId x_node,
                                      const Tensor& x_val, NodeId y_node, const Tensor& y_val);


double train_epoch(Graph& g, std::vector<Param>& params, NodeId x_node, NodeId y_node, NodeId loss_node,
                    const std::vector<NodeId>& grad_nodes, const Dataset& data, Adam& opt);


double compute_accuracy(const Graph& g, const std::vector<Param>& params, NodeId x_node, NodeId y_node,
                         NodeId logits_node, const Dataset& data);

}
