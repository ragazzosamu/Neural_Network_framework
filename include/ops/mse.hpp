#pragma once

#include "core/tensor.hpp"
#include "ops/operation.hpp"

class MseOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Computes the mean squared error between the predicted and target
     *        tensors, row-wise along the last dimension.
     *
     * @return A new Tensor with the same shape as the inputs except the last
     *         dimension, which is collapsed to 1 (one loss value per row),
     *         with this operation registered as its producing operation.
     *
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 2 tensors.
     * @throws std::invalid_argument if either input tensor is null.
     * @throws std::invalid_argument if the predicted and target tensors
     *         don't have the same shape.
     * @throws std::invalid_argument if the predicted tensor has 0 dimensions.
     * @throws std::invalid_argument if the last dimension is 0.
     */
    std::shared_ptr<Tensor> forward() override;

    /**
     * @brief Backpropagates the output gradient, accumulating
     *        d(loss)/d(predicted_j) = grad_row * 2 * (predicted_j - target_j)
     *        into the predicted tensor's gradient (creating it, zero-initialized,
     *        if it doesn't already exist). The target tensor receives no gradient.
     *
     * @param grad Gradient of the loss with respect to this operation's
     *             output; expected to have the same shape as the tensor
     *             returned by forward() (one value per row).
     *
     * @throws std::invalid_argument if `grad` is null.
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 2 tensors.
     * @throws std::invalid_argument if either input tensor is null.
     * @throws std::invalid_argument if the predicted and target tensors
     *         don't have the same shape.
     * @throws std::invalid_argument if the predicted tensor has 0 dimensions.
     * @throws std::invalid_argument if the last dimension is 0.
     *
     * @note This method does not validate that grad's shape actually matches
     *       the (collapsed) loss shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;
};