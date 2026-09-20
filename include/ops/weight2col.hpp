#pragma once

#include "ops/operation.hpp"
#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief Reshapes convolution weights [C_out, C_in, K_H, K_W] into [C_out, K].
 */
class Weight2ColOp : public Operation {
  public:
    using Operation::Operation;

    /**
     * @brief Flattens each output filter into one row.
     * @throws std::invalid_argument if the operation does not have exactly one input.
     * @throws std::invalid_argument if the input tensor is null or does not have shape [C_out, C_in, K_H, K_W].
     */
    std::shared_ptr<Tensor> forward() override;

    /**
     * @brief Restores the column-matrix gradient to the original weight shape.
     * @param grad Gradient with shape [C_out, C_in * K_H * K_W].
     * @throws std::invalid_argument if grad is null.
     * @throws std::invalid_argument if the operation does not have exactly one input.
     * @throws std::invalid_argument if the input tensor is null or does not have shape [C_out, C_in, K_H, K_W].
     * @throws std::invalid_argument if grad does not have the forward output shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;
};