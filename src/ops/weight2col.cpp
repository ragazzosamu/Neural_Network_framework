#include "ops/weight2col.hpp"
#include <stdexcept>

std::shared_ptr<Tensor> Weight2ColOp::forward() {
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("Weight2ColOp expects exactly one input");
    }

    const auto &weight = o_inputs[0];
    if (weight == nullptr) {
        throw std::invalid_argument("Weight tensor must not be null");
    }

    const auto &weight_shape = weight->shape();
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("Weight2ColOp expects weights with shape [C_out, C_in, K_H, K_W]");
    }

    const size_t values_per_filter = weight_shape[1] * weight_shape[2] * weight_shape[3];
    Tensor output({weight_shape[0], values_per_filter});
    const float *weight_data = weight->data();
    float *output_data = output.data();
    const size_t size = weight->size();

    // Weight2Col is simpler than Im2Col: no image positions are visited.
    // There is no output_y/output_x, padding, stride or dilation here because
    // weights are not sliding over an image. The input is already organized as
    // [C_out, C_in, K_H, K_W] and only needs to be viewed as [C_out, K].
    // Each output row is one complete filter. Flattening preserves the order
    // used by Im2ColOp for its rows: channel, kernel_y, then kernel_x.
    // This matching order is required for [C_out, K] @ [K, N].
    for (size_t i = 0; i < size; ++i) {
        output_data[i] = weight_data[i];
    }

    if (weight->requires_grad()) {
        output.set_requires_grad(true);
        output.set_operation(shared_from_this());
    }
    return std::make_shared<Tensor>(std::move(output));
}

void Weight2ColOp::backward(std::shared_ptr<Tensor> grad) const {
    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("Weight2ColOp expects exactly one input");
    }

    const auto &weight = o_inputs[0];
    if (weight == nullptr) {
        throw std::invalid_argument("Weight tensor must not be null");
    }

    const auto &weight_shape = weight->shape();
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("Weight2ColOp expects weights with shape [C_out, C_in, K_H, K_W]");
    }

    const size_t values_per_filter = weight_shape[1] * weight_shape[2] * weight_shape[3];
    if (grad->shape() != std::vector<size_t>{weight_shape[0], values_per_filter}) {
        throw std::invalid_argument("Gradient shape does not match Weight2Col output");
    }

    if (weight->get_grad() == nullptr) {
        weight->set_grad(std::make_shared<Tensor>(weight_shape));
    }

    float *weight_grad = weight->get_grad()->data();
    const float *grad_data = grad->data();
    const size_t size = weight->size();

    // Weight2Col does not move values: it only changes the logical shape.
    // Therefore the inverse operation keeps the same linear index and adds
    // the gradient back to the original [C_out, C_in, K_H, K_W] tensor.
    for (size_t index = 0; index < size; ++index) {
        weight_grad[index] += grad_data[index];
    }
}