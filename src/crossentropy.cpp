#include "crossentropy.hpp"
#include <cmath>
#include <stdexcept>

namespace {
// Added to the predicted probability before taking its log (forward) or
// dividing by it (backward), to avoid log(0) / division by zero when a
// predicted probability underflows to exactly 0.
constexpr float kEpsilon = 1e-7f;
} // namespace

// Computes, for each row, the cross-entropy loss = -sum_j(target_j * log(p_j))
// between the predicted distribution (o_inputs[0], typically a softmax
// output) and the target distribution (o_inputs[1], typically one-hot).
Tensor CrossEntropyOp::forward() {
    if (o_inputs.size() != 2) {
        throw std::invalid_argument("The number of inputs must be 2");
    }

    const auto &model_output = o_inputs[0];
    const auto &target = o_inputs[1];

    if (model_output == nullptr || target == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    if (model_output->shape() != target->shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }

    vector<size_t> shape = model_output->shape();

    // shape.back() below requires at least one dimension.
    if (shape.empty()) {
        throw std::invalid_argument("Input tensors must have at least 1 dimension");
    }

    size_t column_number = shape.back();

    // A zero-length row would make the division below undefined behaviour.
    if (column_number == 0) {
        throw std::invalid_argument("The last dimension must be non-zero");
    }

    auto model_output_data = model_output->data();
    auto target_data = target->data();

    // One loss value per row: same shape as the inputs, but with the last
    // dimension collapsed to 1.
    vector<size_t> loss_shape = shape;
    loss_shape.back() = 1;
    Tensor loss(loss_shape);

    size_t operation_number = model_output->size() / column_number;

    for (size_t i = 0; i < operation_number; ++i) {
        size_t offset = i * column_number;
        float sum = 0.0f;

        // sum_j(target_j * log(p_j)) for this row. Adding kEpsilon guards
        // against log(0): when target_j is 0 (as it is for every non-target
        // class in a one-hot target), that term should contribute 0 to the
        // sum regardless of p_j, but 0 * log(0) would otherwise evaluate to
        // NaN (0 * -inf) in floating point.
        for (size_t j = 0; j < column_number; ++j) {
            sum += target_data[offset + j] * std::log(model_output_data[offset + j] + kEpsilon);
        }

        loss.set_data(i, -sum);
    }

    loss.set_operation(shared_from_this());
    return loss;
}

// Backward pass for loss_i = -sum_j(target_j * log(p_j)): the derivative
// with respect to each predicted probability p_j is d(loss_i)/d(p_j) =
// -target_j / p_j, so the contribution routed back to p_j is
// grad_row * (-target_j / p_j), where grad_row is the incoming gradient for
// the row p_j belongs to. The target tensor is treated as a constant and
// receives no gradient.
void CrossEntropyOp::backward(std::shared_ptr<Tensor> grad) const {
    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }

    if (o_inputs.size() != 2) {
        throw std::invalid_argument("The number of inputs must be 2");
    }

    const auto &model_output = o_inputs[0];
    const auto &target = o_inputs[1];

    if (model_output == nullptr || target == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    if (model_output->shape() != target->shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }

    vector<size_t> shape = model_output->shape();

    if (shape.empty()) {
        throw std::invalid_argument("Input tensors must have at least 1 dimension");
    }

    size_t column_number = shape.back();

    if (column_number == 0) {
        throw std::invalid_argument("The last dimension must be non-zero");
    }

    if (model_output->get_grad() == nullptr) {
        model_output->set_grad(std::make_shared<Tensor>(shape));
    }

    auto model_output_data = model_output->data();
    auto target_data = target->data();
    auto grad_data = grad->data();
    auto grad_output = model_output->get_grad();

    size_t size = model_output->size();

    for (size_t i = 0; i < size; ++i) {
        size_t batch_idx = i / column_number;

        // grad_data holds one value per row (loss has shape (...,1)), so
        // every element within a row shares the same incoming gradient.
        float local_grad = target_data[i] / (model_output_data[i] + kEpsilon);
        float total_grad = grad_data[batch_idx] * local_grad;

        grad_output->add_to_data(i, -total_grad);
    }
}