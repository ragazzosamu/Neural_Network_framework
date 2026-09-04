#include "ops/mse.hpp"
#include <stdexcept>

std::shared_ptr<Tensor> MseOp::forward() {
    if (o_inputs.size() != 2) {
        throw std::invalid_argument("MseOp expects exactly two input tensors");
    }

    auto &predicted = o_inputs[0];
    auto &target = o_inputs[1];

    if (predicted == nullptr || target == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    if (predicted->shape() != target->shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }

    vector<size_t> shape = predicted->shape();

    if (shape.empty()) {
        throw std::invalid_argument("Input tensors must have at least 1 dimension");
    }

    size_t column_number = shape.back();

    if (column_number == 0) {
        throw std::invalid_argument("The last dimension must be non-zero");
    }

    auto predicted_data = predicted->data();
    auto target_data = target->data();

    vector<size_t> loss_shape = shape;
    loss_shape.back() = 1;
    Tensor loss(loss_shape);

    size_t operation_number = predicted->size() / column_number;

    for (size_t i = 0; i < operation_number; ++i) {
        size_t offset = i * column_number;
        float sum = 0.0f;

        for (size_t j = 0; j < column_number; ++j) {
            float diff = predicted_data[offset + j] - target_data[offset + j];
            sum += diff * diff;
        }

        loss.set_data(i, sum / static_cast<float>(column_number));
    }

    auto output = std::make_shared<Tensor>(loss);

    if (predicted->requires_grad() || target->requires_grad()) {
        output->set_requires_grad(true);
        output->set_operation(shared_from_this());
    }
    return output;
}

void MseOp::backward(std::shared_ptr<Tensor> grad) const {
    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }

    if (o_inputs.size() != 2) {
        throw std::invalid_argument("MseOp expects exactly two input tensors");
    }

    auto &predicted = o_inputs[0];
    auto &target = o_inputs[1];

    if (predicted == nullptr || target == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    if (predicted->shape() != target->shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }

    vector<size_t> shape = predicted->shape();

    if (shape.empty()) {
        throw std::invalid_argument("Input tensors must have at least 1 dimension");
    }

    size_t column_number = shape.back();

    if (column_number == 0) {
        throw std::invalid_argument("The last dimension must be non-zero");
    }

    if (predicted->get_grad() == nullptr) {
        predicted->set_grad(std::make_shared<Tensor>(shape));
    }

    auto predicted_data = predicted->data();
    auto target_data = target->data();
    auto grad_data = grad->data();
    auto predicted_grad = predicted->get_grad();

    size_t operation_number = predicted->size() / column_number;

    for (size_t i = 0; i < operation_number; ++i) {
        size_t offset = i * column_number;
        float grad_row = grad_data[i] / static_cast<float>(column_number);

        for (size_t j = 0; j < column_number; ++j) {
            float final_grad = grad_row * 2.0f * (predicted_data[offset + j] - target_data[offset + j]);
            predicted_grad->add_to_data(offset + j, final_grad);
        }
    }
}