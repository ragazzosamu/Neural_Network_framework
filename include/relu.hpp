#include "operation.hpp"
#include <vector>

/**
 * @brief Operation node that computes C = relu(A) = max(0, A), element-wise.
 *
 * The single operand is read from Operation::o_inputs (o_inputs[0] = A).
 * Unlike MatMulOp/MatSumOp, this is a unary operation: it takes exactly one
 * input tensor and involves no broadcasting.
 */
class ReluOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Computes the element-wise ReLU of the input tensor.
     *
     * @return A new Tensor holding C = max(0, A), with this operation
     *         registered as its producing operation (used later by backward()).
     *
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 1 tensor.
     * @throws std::invalid_argument if the input tensor is null.
     */
    Tensor forward() override;

    /**
     * @brief Backpropagates the output gradient through the ReLU, accumulating
     *        it into the input's gradient tensor (creating it, zero-initialized,
     *        if it doesn't already exist). Only the positions where the input
     *        was positive let the gradient through; elsewhere it is blocked.
     *
     * @param grad Gradient of the loss with respect to this operation's output;
     *             expected to have the same shape as the tensor returned by
     *             forward().
     *
     * @throws std::invalid_argument if `grad` is null.
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 1 tensor.
     * @throws std::invalid_argument if the input tensor is null.
     *
     * @note This method does not validate that grad's shape actually matches
     *       the input's shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;
};