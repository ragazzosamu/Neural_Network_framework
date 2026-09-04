#pragma once

#include "core/tensor.hpp"
#include "ops/crossentropy.hpp"
#include "ops/softmax.hpp"

#include <memory>
#include <vector>

/// @brief Computes scalar loss values (cross-entropy, MSE) from a pair of
///        tensors: the model's output and the ground-truth target.
///
/// A Loss object is a thin, stateless wrapper around two input tensors.
/// It does not own or modify them; it only reads their data to produce a
/// single averaged scalar loss.
///
/// @invariant l_inputs always has exactly two elements after construction:
///            l_inputs[0] = model output, l_inputs[1] = target. Both are
///            guaranteed non-null by the constructor.
class Loss {

  public:
    /// @brief Construct a Loss from exactly two tensors.
    /// @param inputs {model_output, target}, in that order.
    /// @throws std::invalid_argument if `inputs` does not contain exactly
    ///         two elements, or if either element is a null pointer.
    explicit Loss(std::vector<std::shared_ptr<Tensor>> inputs);

    /// @brief Softmax cross-entropy loss, averaged over the batch.
    ///
    /// Applies softmax to the model output, feeds the result and the
    /// target into CrossEntropyOp, and averages the resulting per-sample
    /// losses.
    ///
    /// @return The mean cross-entropy loss.
    /// @throws std::invalid_argument if the model output and target sizes
    ///         do not match.
    /// @throws std::runtime_error if the softmax or cross-entropy forward
    ///         pass fails, or if the resulting loss tensor is empty
    ///         (which would otherwise cause a division by zero).
    float cross_entropy();

    /// @brief Mean squared error between model output and target.
    /// @return The mean squared error.
    /// @throws std::invalid_argument if the model output and target sizes
    ///         do not match.
    /// @throws std::runtime_error if the input tensors are empty (which
    ///         would otherwise cause a division by zero).
    float mse();

  private:
    /// Validates that both stored tensors are non-null and have matching
    /// sizes. Shared by cross_entropy() and mse() so both methods fail
    /// the same way on the same bad input.
    /// @throws std::invalid_argument on a null tensor or a size mismatch.
    void validate_inputs() const;

    std::vector<std::shared_ptr<Tensor>> l_inputs; // one is model output the other one target
};