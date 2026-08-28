#include "operation.hpp"
#include <vector>

class MatSumOp : public Operation {
  public:
    using Operation::Operation;

    Tensor forward() const override;
    void backward() const override;

    void broadcast_offsets(size_t i, const vector<size_t> &outShape, const vector<size_t> &stridesA,
                           const vector<size_t> &stridesB, size_t &offsetA, size_t &offsetB);
    vector<size_t> broadcast_strides(const vector<size_t> &strides, const vector<size_t> &shape,
                                     size_t maxRank);
};