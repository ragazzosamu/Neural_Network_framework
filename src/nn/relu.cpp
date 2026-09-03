#include "nn/relu.hpp"
#include "ops/relu.hpp"

std::shared_ptr<Tensor> Relu::forward(const std::shared_ptr<Tensor> &input) const {
    auto relu_op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{input});
    return relu_op->forward();
}