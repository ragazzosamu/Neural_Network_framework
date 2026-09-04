
#include "ops/mean.hpp"
#include <stdexcept>

std::shared_ptr<Tensor> AverageOp::forward() {

    // Check that there is exactly one input.
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("AverageOp expects exactly one input tensor");
    }

    auto &input_tensor = o_inputs[0];

    // Guard against null operands before dereferencing them below.
    if (input_tensor == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    auto const &input_data = input_tensor->data();
    size_t input_size = input_tensor->size();

    // Avoid dividing by zero on an empty tensor.
    if (input_size == 0) {
        throw std::invalid_argument("Input tensor must not be empty");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        sum += input_data[i];
    }

    auto output = std::make_shared<Tensor>(std::vector<size_t>{1});
    output->set_data(0, sum / static_cast<float>(input_size));

    if (input_tensor->requires_grad()) {
        output->set_requires_grad(true);
        output->set_operation(shared_from_this());
    }

    return output;
}

void AverageOp::backward(std::shared_ptr<Tensor> grad) const {

    // Same input count check used in forward().
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("AverageOp expects exactly one input tensor");
    }

    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }

    auto &input_tensor = o_inputs[0];
    size_t input_size = input_tensor->size();

    if (input_tensor->get_grad() == nullptr) {
        input_tensor->set_grad(std::make_shared<Tensor>(input_tensor->shape()));
    }

    auto input_grad = input_tensor->get_grad();
    auto const &grad_data = grad->data();
    float grad_value = grad_data[0] / static_cast<float>(input_size);

    for (size_t i = 0; i < input_size; ++i) {
        input_grad->add_to_data(i, grad_value);
    }
};