#pragma once
#include "module.hpp"
#include "ops/operation.hpp"

class Linear : public Module {

  public:
    Linear(size_t input_size, size_t output_size);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override;
};