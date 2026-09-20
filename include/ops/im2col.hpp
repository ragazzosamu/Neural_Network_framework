#pragma once

#include "ops/operation.hpp"
#include <cstddef>
#include <memory>
#include <vector>

class im2ColOp : public Operation {
  public:
    /**
     * @brief Creates an image-to-column operation for a sliding kernel.
    * @param inputs One image tensor with shape [B, C, H, W].
     * @param kernel_height Kernel height.
     * @param kernel_width Kernel width.
     * @param padding Zero padding applied to both spatial axes.
     * @param kernel_stride Sliding stride applied to both spatial axes.
     * @param dilatation Kernel dilation applied to both spatial axes.
     */
    im2ColOp(std::vector<std::shared_ptr<Tensor>> inputs, size_t kernel_height, size_t kernel_width, size_t padding = 0, size_t kernel_stride = 1,
             size_t dilatation = 1);

    /**
     * @brief Converts sliding image patches into rows of a column matrix.
     * @throws std::invalid_argument if the operation does not have exactly one input.
     * @throws std::invalid_argument if a kernel dimension, stride, or dilation is zero.
    * @throws std::invalid_argument if the input tensor is null or does not have shape [B, C, H, W].
     * @throws std::invalid_argument if the effective kernel is larger than the padded input.
     */
    std::shared_ptr<Tensor> forward() override;

    /**
     * @brief Accumulates the column-matrix gradient back into the image.
     * @param grad Gradient with the same shape as the forward output.
     * @throws std::invalid_argument if grad is null.
     * @throws std::invalid_argument if the operation does not have exactly one input.
     * @throws std::invalid_argument if a kernel dimension, stride, or dilation is zero.
    * @throws std::invalid_argument if the input tensor is null or does not have shape [B, C, H, W].
     * @throws std::invalid_argument if the effective kernel is larger than the padded input.
     * @throws std::invalid_argument if grad does not have the forward output shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;

  private:
    size_t kernel_height;
    size_t kernel_width;
    size_t padding;
    size_t kernel_stride;
    size_t dilatation;

    std::shared_ptr<Tensor> process(bool backward, const std::shared_ptr<Tensor> &grad) const;
};