#include "operation.hpp"
#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief Reshapes column matrices [B, C_out, H_out * W_out] into [B, C_out, H_out, W_out].
 */
class col2ImOp : public Operation {
  public:
    /**
     * @brief Creates a column-to-image operation.
    * @param inputs One tensor with shape [B, C_out, H_out * W_out].
     * @param output_height Height of the output feature map.
     * @param output_width Width of the output feature map.
     */
    col2ImOp(std::vector<std::shared_ptr<Tensor>> inputs, size_t output_height, size_t output_width);

    /**
     * @brief Reshapes the input column matrix into a feature map tensor.
     * @throws std::invalid_argument if the operation does not have exactly one input.
     * @throws std::invalid_argument if an output dimension is zero.
    * @throws std::invalid_argument if the input tensor is null, has the wrong rank,
    *         or its last dimension does not equal H_out * W_out.
     */
    std::shared_ptr<Tensor> forward() override;

    /**
     * @brief Flattens the output gradient and accumulates it into the input.
    * @param grad Gradient with shape [B, C_out, H_out, W_out].
     * @throws std::invalid_argument if grad is null.
     * @throws std::invalid_argument if the operation does not have exactly one input.
     * @throws std::invalid_argument if an output dimension is zero.
    * @throws std::invalid_argument if the input tensor is null, has the wrong rank,
    *         or its last dimension does not equal H_out * W_out.
     * @throws std::invalid_argument if grad does not have the forward output shape.
     */
    void backward(std::shared_ptr<Tensor> grad) const override;

  private:
    size_t output_height;
    size_t output_width;
};