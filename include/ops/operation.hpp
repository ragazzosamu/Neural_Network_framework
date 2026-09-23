#pragma once
#include "core/tensor.hpp"
#include <memory>
#include <vector>

class Tensor; // Forward declaration di Tensor
class Operation : public std::enable_shared_from_this<Operation> {
  public:
    virtual ~Operation() = default;
    explicit Operation(std::vector<std::shared_ptr<Tensor>> inputs) : o_inputs(std::move(inputs)) {}

    virtual std::shared_ptr<Tensor> forward() = 0;

    virtual void backward(std::shared_ptr<Tensor> grad) const = 0;

    virtual const std::vector<std::shared_ptr<Tensor>> &inputs() const { return o_inputs; }

  protected:
    virtual bool is_contiguous_2d(const std::shared_ptr<Tensor> &tensor) const {
        const auto &shape = tensor->shape();
        const auto &strides = tensor->strides();

        return shape.size() == 2 && strides.size() == 2 && strides[1] == 1 && strides[0] == shape[1];
    }

    std::vector<std::shared_ptr<Tensor>> o_inputs;
};