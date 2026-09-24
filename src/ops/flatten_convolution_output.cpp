#include "ops/flatten_convolution_output.hpp"

#include <stdexcept>
#include <vector>

FlattenConvolutionOutputOp::FlattenConvolutionOutputOp(const std::shared_ptr<Tensor> &input) : Operation({input}) {}

std::shared_ptr<Tensor> FlattenConvolutionOutputOp::forward() {
    const auto &input = o_inputs[0];
    if (!input || input->shape().size() != 4) {
        throw std::invalid_argument("FlattenConvolutionOutputOp expects input shaped [N, C, H, W]");
    }

    const auto &shape = input->shape();
    Tensor flattened = input->clone();
    flattened.reshape(std::vector<size_t>{shape[0], shape[1] * shape[2] * shape[3]});
    auto output = std::make_shared<Tensor>(flattened);
    if (input->requires_grad()) {
        output->set_requires_grad(true);
        output->set_operation(shared_from_this());
    }
    return output;
}

void FlattenConvolutionOutputOp::backward(std::shared_ptr<Tensor> grad) const {
    const auto &input = o_inputs[0];
    if (!grad || !input) {
        throw std::invalid_argument("FlattenConvolutionOutputOp requires non-null input and gradient");
    }
    if (input->get_grad() == nullptr) {
        input->set_grad(std::make_shared<Tensor>(input->shape()));
    }

    float *input_grad = input->get_grad()->data();
    const float *grad_data = grad->data();
    const std::size_t size = input->size();
    for (std::size_t index = 0; index < size; ++index) {
        input_grad[index] += grad_data[index];
    }
}