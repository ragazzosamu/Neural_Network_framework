#include "nn/relu.hpp"
#include "ops/relu.hpp"

// Delegates the actual element-wise computation to ReluOp, keeping Relu
// itself as a thin wrapper: Module gives it parameters()/modules()/
// set_train()/set_eval() for free, while ReluOp owns the math.
//
// No try/catch here (unlike Linear::forward): whatever ReluOp::forward()
// throws — e.g. for a null input — propagates to the caller as-is.
std::shared_ptr<Tensor> Relu::forward(const std::shared_ptr<Tensor> &input) const {
    auto op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{input});
    return op->forward();
}