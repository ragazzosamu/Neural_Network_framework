#pragma once

#include "operation.hpp"

/**
 * @brief Flattens convolution output tensors from [N, C, H, W] to [N, C*H*W].
 */
class FlattenConvolutionOutputOp : public Operation {
  public:
    explicit FlattenConvolutionOutputOp(const std::shared_ptr<Tensor> &input);

    /**
     * @brief Flattens the convolution output while preserving autograd links.
     */
    std::shared_ptr<Tensor> forward() override;

    /**
     * @brief Reshapes and accumulates the output gradient into the input.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;
};