#include "operation.hpp"
#include <vector>

/**
 * @brief Operation node that computes C = A @ B (matrix multiplication) with
 *        NumPy-style broadcasting on every dimension except the last two.
 *
 * The two operands are read from Operation::o_inputs (o_inputs[0] = A,
 * o_inputs[1] = B). The last two dimensions of each tensor are treated as the
 * actual matrix rows/columns; all leading dimensions are batch dimensions and
 * follow standard broadcasting rules (equal, or one of the two equal to 1).
 */
class MatMulOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Computes the (broadcasted) matrix product of the two input tensors.
     *
     * @return A new Tensor holding C = A @ B, with this operation registered
     *         as its producing operation (used later by backward()).
     *
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 2 tensors.
     * @throws std::invalid_argument if either input tensor is null.
     * @throws std::invalid_argument if both A and B have fewer than 2 dimensions
     *         (at least one of the two must be a matrix, or higher-rank tensor).
     * @throws std::invalid_argument if A's last dimension does not match B's
     *         second-to-last dimension (incompatible for matrix multiplication).
     * @throws std::invalid_argument if any leading (batch) dimension of A and B
     *         is neither equal nor equal to 1 (incompatible for broadcasting).
     * @throws std::invalid_argument if A's row count or B's column count is 0.
     */
    Tensor forward() override;

    /**
     * @brief Backpropagates the output gradient through the matrix multiplication,
     *        accumulating dA = grad @ B^T and dB = A^T @ grad into A's and B's
     *        gradient tensors (creating them, zero-initialized, if they don't
     *        already exist).
     *
     * @param grad Gradient of the loss with respect to this operation's output;
     *             expected to have the same (broadcast) shape as the tensor
     *             returned by forward().
     *
     * @throws std::invalid_argument if `grad` is null.
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 2 tensors.
     * @throws std::invalid_argument if either input tensor is null.
     * @throws std::invalid_argument if both A and B have fewer than 2 dimensions
     *         (at least one of the two must be a matrix, or higher-rank tensor).
     * @throws std::invalid_argument if grad's row count or column count is 0.
     *
     * @note This method still does not validate that grad's shape is actually
     *       compatible with A and B's (broadcast) shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;

  private:
    /**
     * @brief Decomposes a flat batch index into per-dimension coordinates and,
     *        together with each tensor's broadcast-adjusted strides, computes
     *        the memory offset of that batch's matrix in A, B and the output.
     *
     * @param i             Flat batch index, expected in [0, number_of_batches).
     * @param outShape      Shape of the (broadcast) output tensor.
     * @param stridesA      Broadcast-adjusted strides of A (see broadcast_strides()).
     * @param stridesB      Broadcast-adjusted strides of B.
     * @param stridesOutput Broadcast-adjusted strides of the output.
     * @param[out] offsetA      Memory offset of batch i within A.
     * @param[out] offsetB      Memory offset of batch i within B.
     * @param[out] offsetOutput Memory offset of batch i within the output.
     */
    static void broadcast_offsets(size_t i, const std::vector<size_t> &outShape, const std::vector<size_t> &stridesA,
                                  const std::vector<size_t> &stridesB, const std::vector<size_t> &stridesOutput, size_t &offsetA, size_t &offsetB,
                                  size_t &offsetOutput);

    /**
     * @brief Pads `strides` on the left with 0s up to `maxRank`, then zeroes the
     *        stride of every axis where `shape` is 1 (a broadcast axis), so that
     *        iterating along a broadcast axis always re-reads the same element.
     *
     * @param strides Original strides of the tensor.
     * @param shape   Shape of the tensor, already padded to `maxRank` dimensions.
     * @param maxRank Target rank to broadcast to.
     * @return The broadcast-adjusted strides, of size maxRank.
     */
    static std::vector<size_t> broadcast_strides(const std::vector<size_t> &strides, const std::vector<size_t> &shape, size_t maxRank);
};