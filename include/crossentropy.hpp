#pragma once
#include "operation.hpp"
#include <memory>
#include <vector>

/**
 * @brief Operation node that computes the (row-wise) cross-entropy loss
 *        between a predicted distribution and a one-hot target distribution.
 *
 * Reads two operands from Operation::o_inputs: o_inputs[0] is the predicted
 * distribution (typically the output of a SoftmaxOp), o_inputs[1] is the
 * target distribution. Both are expected to share the same shape; the loss
 * is computed independently for each row along the last dimension, exactly
 * like SoftmaxOp, and collapses that dimension to size 1 in the output.
 */
class CrossEntropyOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Computes, for each row, loss = -sum_j(target_j * log(p_j)).
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
    Tensor forward() override;

    /**
     * @brief Backpropagates the output gradient, accumulating
     *        d(loss)/d(p_j) = -grad_row * target_j / p_j into the predicted
     *        tensor's gradient (creating it, zero-initialized, if it doesn't
     *        already exist). The target tensor receives no gradient.
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