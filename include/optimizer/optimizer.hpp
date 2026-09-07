#pragma once

#include "core/tensor.hpp"

/**
 * @brief Base class for all optimization algorithms.
 *
 * An optimizer owns the learning rate and the list of trainable parameters it
 * updates during training. Concrete subclasses implement the actual update rule
 * in @ref step(), while @ref zero_grad() clears each parameter gradient before
 * the next backward pass.
 */
class Optimizer {

  public:
    /**
     * @brief Constructs an optimizer for the provided parameters.
     *
     * @param learning_rate Scaling factor applied to every parameter update.
     * @param params Trainable tensors that will be optimized.
     */
    explicit Optimizer(float learning_rate, std::vector<std::shared_ptr<Tensor>> params) : learning_rate(learning_rate), params(std::move(params)) {}

    virtual ~Optimizer() = default;

    /**
     * @brief Clears the stored gradient for every optimized parameter.
     *
     */
    virtual void zero_grad() {
        for (auto &param : params) {
            param->set_grad(nullptr);
        }
    }

    /**
     * @brief Performs one optimization step.
     *
     * Derived classes implement the actual update logic for the chosen
     * optimization method.
     */
    virtual void step() = 0;

  protected:
    /// Learning rate used by the optimizer update rule.
    float learning_rate;

    /// Parameters updated by this optimizer during training.
    std::vector<std::shared_ptr<Tensor>> params;
};