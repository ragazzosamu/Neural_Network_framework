#pragma once
#include "module.hpp"

class Relu : public Module {

  public:
    Relu() = default;

    /**
     * @brief Applies the ReLU function element-wise to the input tensor.
     *
     * @param input The input tensor.
     * @return A new Tensor with the same shape as @p input, where each
     *         element is max(0, x).
     *
     * @throws Whatever exception ReluOp's constructor or forward() throws
     *         (e.g. for a null input); Relu::forward() does not catch or
     *         wrap it, unlike Linear::forward().
     */
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override;
};