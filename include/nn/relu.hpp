#pragma once
#include "module.hpp"

class Relu : public Module {

  public:
    Relu() = default;

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override;
};