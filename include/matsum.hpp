#pragma once
#include "operation.hpp"
#include <memory>
#include <vector>

/**
 * @brief Operation node that computes C = A + B (element-wise sum) with
 *        NumPy-style broadcasting on every dimension.
 *
 * The two operands are read from Operation::o_inputs (o_inputs[0] = A,
 * o_inputs[1] = B). Unlike MatMulOp, there is no special treatment of the
 * last two dimensions: every axis follows the same broadcasting rules
 * (equal, or one of the two equal to 1).
 */
class MatSumOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Computes the (broadcasted) element-wise sum of the two input tensors.
     *
     * @return A new Tensor holding C = A + B, with this operation registered
     *         as its producing operation (used later by backward()).
     *
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 2 tensors.
     * @throws std::invalid_argument if either input tensor is null.
     * @throws std::invalid_argument if any dimension of A and B is neither
     *         equal nor equal to 1 (incompatible for broadcasting).
     */
    Tensor forward() override;

    /**
     * @brief Backpropagates the output gradient through the sum, accumulating
     *        it into A's and B's gradient tensors (creating them,
     *        zero-initialized, if they don't already exist). Any axis that
     *        was broadcast during forward() is summed back over here.
     *
     * @param grad Gradient of the loss with respect to this operation's output;
     *             expected to have the same (broadcast) shape as the tensor
     *             returned by forward().
     *
     * @throws std::invalid_argument if `grad` is null.
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 2 tensors.
     * @throws std::invalid_argument if either input tensor is null.
     *
     * @note This method does not validate that grad's shape is actually
     *       compatible with A and B's (broadcast) shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;

  private:
    /**
     * @brief Given a flat output index, decomposes it into per-dimension
     *        coordinates and uses them, together with each tensor's
     *        broadcast-adjusted strides, to compute the corresponding memory
     *        offset in A and B.
     *
     * @param i        Flat output index, expected in [0, output.size()).
     * @param outShape Shape of the (broadcast) output tensor.
     * @param stridesA Broadcast-adjusted strides of A (see broadcast_strides()).
     * @param stridesB Broadcast-adjusted strides of B.
     * @param[out] offsetA Memory offset of element i within A.
     * @param[out] offsetB Memory offset of element i within B.
     */
    static void broadcast_offsets(size_t i, const std::vector<size_t> &outShape, const std::vector<size_t> &stridesA,
                                  const std::vector<size_t> &stridesB, size_t &offsetA, size_t &offsetB);

    /**
     * @brief Pads `strides` on the left with 0s up to `maxRank`, then zeroes
     *        the stride of every axis where `shape` is 1 (a broadcast axis),
     *        so that iterating along a broadcast axis always re-reads/re-writes
     *        the same element.
     *
     * @param strides Original strides of the tensor.
     * @param shape   Shape of the tensor, already padded to `maxRank` dimensions.
     * @param maxRank Target rank to broadcast to.
     * @return The broadcast-adjusted strides, of size maxRank.
     */
    static std::vector<size_t> broadcast_strides(const std::vector<size_t> &strides, const std::vector<size_t> &shape, size_t maxRank);
};