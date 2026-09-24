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

    // True if the tensor is laid out row-major with no gaps, whatever its rank,
    // so that element i of its logical (row-major) order is data()[i].
    virtual bool is_contiguous(const std::shared_ptr<Tensor> &tensor) const {
        const auto &shape = tensor->shape();
        const auto &strides = tensor->strides();

        size_t expected = 1;
        for (size_t i = shape.size(); i-- > 0;) {
            if (strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    std::vector<std::shared_ptr<Tensor>> o_inputs;
};