#pragma once
#include "tensor.hpp"
#include <memory>
#include <vector>

class Tensor; // Forward declaration di Tensor
class Operation : public std::enable_shared_from_this<Operation> {
  public:
    virtual ~Operation() = default;
    explicit Operation(std::vector<std::shared_ptr<Tensor>> inputs) : o_inputs(std::move(inputs)) {}

    virtual Tensor forward() = 0;

    virtual void backward(std::shared_ptr<Tensor> grad) const = 0;

    virtual std::vector<std::shared_ptr<Tensor>> inputs() const { return o_inputs; }

  protected:
    std::vector<std::shared_ptr<Tensor>> o_inputs;
};