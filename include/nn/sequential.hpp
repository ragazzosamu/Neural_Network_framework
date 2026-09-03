#pragma once
#include "nn/module.hpp"

class Sequential : public Module {

  public:
    explicit Sequential(const std::vector<std::shared_ptr<Module>> layers);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override;
};