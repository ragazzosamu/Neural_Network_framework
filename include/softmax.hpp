#include "operation.hpp"
#include <vector>

/**
 * @brief Operation node that computes softmax(A) along the last dimension.
 *
 * The single operand is read from Operation::o_inputs (o_inputs[0] = A).
 * Softmax is applied independently to each "row" of the last dimension: for
 * an input of shape (..., row_length), every group of row_length elements is
 * turned into a probability distribution that sums to 1. In the common case
 * A is a single row (a vector); for a transformer it is typically a matrix
 * (or higher-rank tensor).
 */
class SoftmaxOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Computes softmax(A) row-wise along the last dimension.
     *
     * As a side effect, caches the result in `saved_output`, since backward()
     * needs it to compute the gradient.
     *
     * @return A new Tensor holding softmax(A), with this operation registered
     *         as its producing operation (used later by backward()).
     *
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 1 tensor.
     * @throws std::invalid_argument if the input tensor is null.
     * @throws std::invalid_argument if the input tensor has 0 dimensions.
     * @throws std::invalid_argument if the input's last dimension is 0.
     */
    Tensor forward() override;

    /**
     * @brief Backpropagates the output gradient through the softmax,
     *        accumulating it into the input's gradient tensor (creating it,
     *        zero-initialized, if it doesn't already exist).
     *
     * For each row, with y = saved_output and g = grad, computes
     * dx_j = y_j * (g_j - sum_k(g_k * y_k)), following the softmax Jacobian.
     *
     * @param grad Gradient of the loss with respect to this operation's output;
     *             expected to have the same shape as the tensor returned by
     *             forward().
     *
     * @throws std::invalid_argument if `grad` is null.
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 1 tensor.
     * @throws std::invalid_argument if the input tensor is null.
     * @throws std::invalid_argument if called before forward() has ever run
     *         (no cached output available yet).
     * @throws std::invalid_argument if the input tensor has 0 dimensions.
     * @throws std::invalid_argument if the input's last dimension is 0.
     *
     * @note This method does not validate that grad's shape actually matches
     *       the input's shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;

  private:
    /// Cached output of the last forward() call, y = softmax(x); needed by
    /// backward() to compute the gradient (see the formula above).
    std::shared_ptr<Tensor> saved_output;
};