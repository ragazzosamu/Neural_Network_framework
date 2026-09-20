#include "module.hpp"
#include "ops/col2im.hpp"
#include "ops/im2col.hpp"
#include "ops/matmul.hpp"
#include "ops/weight2col.hpp"

/**
 * @brief Two-dimensional convolution implemented with im2col and matrix multiplication.
 *
 * The input has shape [B, C_in, H, W]. The layer stores weights with shape
 * [C_out, C_in, K_H, K_W] and one bias per output channel with shape
 * [C_out, 1]. The output has shape [B, C_out, H_out, W_out].
 */
class Conv2D : public Module {
  public:
    /**
     * @brief Creates and initializes a 2D convolution layer.
     *
     * Weights use a standard normal distribution and biases are initialized
     * to zero.
     *
     * @param C_in Number of input channels.
     * @param C_out Number of output channels and filters.
     * @param kernel_height Kernel height.
     * @param kernel_width Kernel width.
     * @param padding Zero padding applied to both spatial axes.
     * @param stride Sliding stride applied to both spatial axes.
     * @param dilatation Kernel dilation applied to both spatial axes.
     */
    Conv2D(size_t C_in, size_t C_out, size_t kernel_height, size_t kernel_width, size_t padding = 0, size_t stride = 1, size_t dilatation = 1);

    /**
     * @brief Computes the convolution output.
    * @param input Input tensor with shape [B, C_in, H, W].
    * @return Tensor with shape [B, C_out, H_out, W_out].
     * @throws std::runtime_error if the input is null or any internal
     *         operation receives incompatible shapes. The original exception
     *         is preserved as a nested exception.
     */
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override;

  private:
    size_t kernel_height;
    size_t kernel_width;
    size_t padding;
    size_t stride;
    size_t dilatation;
};