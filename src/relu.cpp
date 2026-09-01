#include "relu.hpp"
#include <stdexcept>

// Computes C = relu(A) = max(0, A), element-wise.
Tensor ReluOp::forward() {
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("The number of inputs must be 1");
    }

    const auto &input = o_inputs[0];

    // Guard against a null operand before dereferencing it below.
    if (input == nullptr) {
        throw std::invalid_argument("Input tensor must not be null");
    }

    auto data = input->data();

    Tensor output(input->shape());

    // The Tensor constructor already zero-initializes `output`, so only the
    // positive elements need to be written explicitly; everywhere data[i] <= 0
    // the output is left at its default 0.
    for (size_t i = 0; i < input->size(); ++i) {
        if (data[i] > 0.0f) {
            output.set_data(i, data[i]);
        }
    }

    output.set_operation(shared_from_this());
    return output;
}

// Backward pass for C = relu(A): the derivative of relu is 1 where the input
// was positive and 0 elsewhere, so the incoming gradient is passed straight
// through to A wherever A was positive, and blocked (contributes nothing)
// wherever A was zero or negative.
void ReluOp::backward(std::shared_ptr<Tensor> grad) const {
    // Guard against a null gradient before dereferencing it below.
    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }

    if (o_inputs.size() != 1) {
        throw std::invalid_argument("The number of inputs must be 1");
    }

    const auto &input = o_inputs[0];

    // Guard against a null operand before dereferencing it below.
    if (input == nullptr) {
        throw std::invalid_argument("Input tensor must not be null");
    }

    auto data = input->data();

    if (input->get_grad() == nullptr) {
        input->set_grad(std::make_shared<Tensor>(input->shape()));
    }

    auto input_grad = input->get_grad();
    auto grad_data = grad->data();

    for (size_t i = 0; i < input->size(); ++i) {
        // Route the gradient through only where the input was positive
        // (the "gate" left open by relu's derivative).
        if (data[i] > 0.0f) {
            input_grad->add_to_data(i, grad_data[i]);
        }
    }
}