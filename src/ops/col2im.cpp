#include "ops/col2im.hpp"
#include <stdexcept>

col2ImOp::col2ImOp(std::vector<std::shared_ptr<Tensor>> inputs, size_t output_height, size_t output_width)
    : Operation(std::move(inputs)), output_height(output_height), output_width(output_width) {}

std::shared_ptr<Tensor> col2ImOp::forward() {
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("Col2ImOp expects exactly one input");
    }
    if (output_height == 0 || output_width == 0) {
        throw std::invalid_argument("Col2ImOp output dimensions must be non-zero");
    }

    const auto &input = o_inputs[0];
    if (input == nullptr) {
        throw std::invalid_argument("Input tensor must not be null");
    }
    if (input->shape().size() != 3) {
        throw std::invalid_argument("Col2ImOp expects an input with shape [B, C_out, H_out * W_out]");
    }
    if (input->shape()[2] != output_height * output_width) {
        throw std::invalid_argument("Input shape does not match Col2ImOp output dimensions");
    }

    auto output = std::make_shared<Tensor>(std::vector<size_t>{input->shape()[0], input->shape()[1], output_height, output_width});
    const auto input_data = input->data();
    for (size_t i = 0; i < input->size(); ++i) {
        output->set_data(i, input_data[i]);
    }

    if (input->requires_grad()) {
        output->set_requires_grad(true);
        output->set_operation(shared_from_this());
    }
    return output;
}

void col2ImOp::backward(std::shared_ptr<Tensor> grad) const {
    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("Col2ImOp expects exactly one input");
    }

    const auto &input = o_inputs[0];
    if (input == nullptr) {
        throw std::invalid_argument("Input tensor must not be null");
    }
    if (input->shape().size() != 3 || output_height == 0 || output_width == 0 || input->shape()[2] != output_height * output_width) {
        throw std::invalid_argument("Input shape does not match Col2ImOp output dimensions");
    }
    if (grad->shape() != std::vector<size_t>{input->shape()[0], input->shape()[1], output_height, output_width}) {
        throw std::invalid_argument("Gradient shape does not match Col2ImOp output");
    }

    if (input->get_grad() == nullptr) {
        input->set_grad(std::make_shared<Tensor>(input->shape()));
    }

    const auto grad_data = grad->data();
    auto input_grad = input->get_grad();
    for (size_t i = 0; i < input->size(); ++i) {
        input_grad->add_to_data(i, grad_data[i]);
    }
}
