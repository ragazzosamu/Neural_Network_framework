#pragma once

#include "ops/operation.hpp"

class AverageOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Computes the average of all elements in the input tensor.
     *
     * @return A new Tensor of shape {1} holding the mean of the input
     *         tensor's elements, with this operation registered as its
     *         producing operation (used later by backward()).
     *
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 1 tensor.
     * @throws std::invalid_argument if the input tensor is null.
     * @throws std::invalid_argument if the input tensor is empty (size 0).
     */
    Tensor forward() override;

    /**
     * @brief Distributes the incoming gradient equally across all elements
     *        of the input tensor.
     *
     * @param grad Gradient tensor of shape {1} flowing back from the output.
     *
     * @throws std::invalid_argument if Operation::o_inputs does not contain
     *         exactly 1 tensor.
     * @throws std::invalid_argument if grad is null.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;
};