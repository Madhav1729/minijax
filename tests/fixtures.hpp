#pragma once


#include "minijax/ir.hpp"

namespace minijax::test_fixtures {


struct LossFixture {
    Graph g;
    NodeId W, x, y, loss;
};

inline LossFixture make_loss_fixture() {
    LossFixture f;
    f.W = f.g.input({2, 2});
    f.x = f.g.input({2, 1});
    f.y = f.g.input({2, 1});
    NodeId pred = f.g.relu(f.g.matmul(f.W, f.x));
    NodeId diff = f.g.sub(pred, f.y);
    f.loss = f.g.sum(diff);
    return f;
}

}
