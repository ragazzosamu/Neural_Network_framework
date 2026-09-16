#pragma once

#include "core/tensor.hpp"
#include "ops/crossentropy.hpp"
#include "ops/mean.hpp"
#include "ops/mse.hpp"
#include "ops/softmax.hpp"

#include <memory>
#include <vector>

/// @brief Computes scalar loss values (cross-entropy, MSE) from a pair of
///        tensors: the model's output and the ground-truth target.
///
/// Loss exposes stateless operations that do not require an object instance.
class Loss {

  public:
    Loss() = delete;

    /// @brief Softmax cross-entropy loss, averaged over the batch.
    ///
    /// Applies softmax to the model output, feeds the result and the
    /// target into CrossEntropyOp, and averages the resulting per-sample
    /// losses.
    ///
    /// @param  model_output model prediction
    /// @param  target ground truth
    /// @return The mean cross-entropy loss.
    /// @throws std::invalid_argument if the model output and target sizes
    ///         do not match.
    /// @throws std::runtime_error if the softmax or cross-entropy forward
    ///         pass fails, or if the resulting loss tensor is empty
    ///         (which would otherwise cause a division by zero).
    static std::shared_ptr<Tensor> cross_entropy(std::shared_ptr<Tensor> &model_output, std::shared_ptr<Tensor> &target);

    /// @brief  Computes the (row-wise) mean squared error
    ///         between a row of predicted values and a row of target values.
    ///
    /// @param  model_output model prediction
    /// @param  target ground truth
    /// @return The mean squared error.
    /// @throws std::invalid_argument if the model output and target sizes
    ///         do not match.
    /// @throws std::runtime_error if the input tensors are empty (which
    ///         would otherwise cause a division by zero).
    static std::shared_ptr<Tensor> mse(std::shared_ptr<Tensor> &model_output, std::shared_ptr<Tensor> &target);

  private:
    /// @param  model_output model prediction
    /// @param  target ground truth
    /// Validates that both tensors are non-null and have matching sizes.
    /// Shared by cross_entropy() and mse() so both methods fail
    /// the same way on the same bad input.
    /// @throws std::invalid_argument on a null tensor or a size mismatch.
    static void validate_inputs(std::shared_ptr<Tensor> &model_output, std::shared_ptr<Tensor> &target);
};