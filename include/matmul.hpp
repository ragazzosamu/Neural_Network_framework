#include "operation.hpp"
#include <vector>

class MatMulOp : public Operation {
  public:
    using Operation::Operation;

    Tensor forward() const override;
    void backward() const override;
};