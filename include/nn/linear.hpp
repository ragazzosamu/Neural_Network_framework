#pragma once
#include "module.hpp"
#include "ops/operation.hpp"

/**
 * @brief Fully-connected (dense) layer: computes x * W + b.
 *
 * Holds two learnable parameters, registered in Module::params so they are
 * picked up automatically by parameters() / set_train() / set_eval():
 * - weights, with shape {input_size, output_size}
 * - biases, with shape {output_size, 1}
 *
 * @note This layer follows the common "input as a row vector" convention
 *       (output = x * W + b), not the "input as a column vector"
 *       convention (output = W * x + b).
 */
class Linear : public Module {

  public:
    /**
     * @brief Constructs a Linear layer and initializes its parameters.
     *
     * Weights are drawn from a standard normal distribution (mean 0,
     * stddev 1); biases are initialized to zero.
     *
     * @param input_size Number of input features (rows of the weight matrix).
     * @param output_size Number of output features (columns of the weight
     *        matrix, and the length of the bias vector).
     */
    Linear(size_t input_size, size_t output_size);

    /**
     * @brief Computes input * W + b.
     *
     * @param input The input tensor, expected to be shape-compatible with
     *        the weight matrix under the layer's row-vector convention.
     * @return A new Tensor holding the layer's output.
     *
     * @throws std::runtime_error if the underlying matrix multiplication or
     *         addition fails (e.g. incompatible shapes, null input). The
     *         original exception is preserved as a nested exception and can
     *         be recovered with std::rethrow_if_nested.
     */
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override;
};