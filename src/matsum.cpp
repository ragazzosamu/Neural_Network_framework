#include "matsum.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

// Computes C = A + B element-wise, with NumPy-style broadcasting on every
// dimension (unlike MatMulOp, there's no "special" last-two-dimensions
// treatment here: every axis is broadcast the same way).
Tensor MatSumOp::forward() {
    if (o_inputs.size() != 2) {
        throw std::invalid_argument("The number of inputs must be 2");
    }

    const auto &tensorA = o_inputs[0];
    const auto &tensorB = o_inputs[1];

    // Guard against null operands before dereferencing them below.
    if (tensorA == nullptr || tensorB == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    std::vector<size_t> shapeA = tensorA->shape();
    std::vector<size_t> shapeB = tensorB->shape();

    size_t maxRank = std::max(shapeA.size(), shapeB.size());

    // Pad the shorter shape with leading 1s so both tensors have the same rank
    // (this is what makes the broadcasting rules below simple to apply).
    shapeA.insert(shapeA.begin(), maxRank - shapeA.size(), 1);
    shapeB.insert(shapeB.begin(), maxRank - shapeB.size(), 1);

    // Every dimension follows standard broadcasting rules: equal, or one of the
    // two is 1. The output takes the larger of the two sizes on each axis.
    std::vector<size_t> shapeOutput(maxRank);
    for (size_t i = 0; i < maxRank; ++i) {
        if (shapeA[i] != shapeB[i] && shapeA[i] != 1 && shapeB[i] != 1) {
            throw std::invalid_argument("The two shapes are not compatible for broadcasting");
        }
        shapeOutput[i] = std::max(shapeA[i], shapeB[i]);
    }

    Tensor output(shapeOutput);
    size_t output_size = output.size();

    if (shapeA == shapeB) {
        // Fast path: no broadcasting is actually needed (both operands already
        // have the exact same shape), so both tensors can be walked with a
        // single flat index, without going through the offset machinery below.
        for (size_t i = 0; i < output_size; ++i) {
            output.set_data(i, tensorA->data()[i] + tensorB->data()[i]);
        }
    } else {
        // General path: at least one axis is being broadcast, so A and B each
        // need their own (possibly repeated) offset for every output element.
        std::vector<size_t> stridesA = broadcast_strides(tensorA->strides(), shapeA, maxRank);
        std::vector<size_t> stridesB = broadcast_strides(tensorB->strides(), shapeB, maxRank);

        for (size_t i = 0; i < output_size; ++i) {
            size_t offsetA, offsetB;
            broadcast_offsets(i, shapeOutput, stridesA, stridesB, offsetA, offsetB);

            output.set_data(i, tensorA->data()[offsetA] + tensorB->data()[offsetB]);
        }
    }

    output.set_operation(shared_from_this());
    return output;
}

// Backward pass for C = A + B: the local derivative of a sum with respect to
// each operand is 1, so the incoming gradient is simply routed to A and B
// unchanged, element by element — except that any axis A or B was broadcast
// over must be summed back over. Accumulating the same grad element into
// every broadcast position (via add_to_data, i.e. +=) is exactly what
// "collapses" that axis back down, which is the correct gradient for a
// broadcast sum.
void MatSumOp::backward(std::shared_ptr<Tensor> grad) const {
    // Guard against a null gradient before dereferencing it below.
    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }

    if (o_inputs.size() != 2) {
        throw std::invalid_argument("The number of inputs must be 2");
    }

    const auto &tensorA = o_inputs[0];
    const auto &tensorB = o_inputs[1];

    // Guard against null operands before dereferencing them below.
    if (tensorA == nullptr || tensorB == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    std::vector<size_t> shapeA = tensorA->shape();
    if (tensorA->get_grad() == nullptr) {
        tensorA->set_grad(std::make_shared<Tensor>(shapeA));
    }

    std::vector<size_t> shapeB = tensorB->shape();
    if (tensorB->get_grad() == nullptr) {
        tensorB->set_grad(std::make_shared<Tensor>(shapeB));
    }

    size_t maxRank = std::max(shapeA.size(), shapeB.size());

    shapeA.insert(shapeA.begin(), maxRank - shapeA.size(), 1);
    shapeB.insert(shapeB.begin(), maxRank - shapeB.size(), 1);

    // Broadcast-adjusted strides: on a broadcast axis (size 1), the stride is
    // zeroed, so every position along that axis maps back to the same element
    // of A/B — this is what makes add_to_data() below accumulate (sum) the
    // gradient over that axis instead of overwriting it.
    std::vector<size_t> stridesA = broadcast_strides(tensorA->strides(), shapeA, maxRank);
    std::vector<size_t> stridesB = broadcast_strides(tensorB->strides(), shapeB, maxRank);

    for (size_t i = 0; i < grad->size(); ++i) {
        size_t offsetA, offsetB;
        broadcast_offsets(i, grad->shape(), stridesA, stridesB, offsetA, offsetB);

        // Route (and, on broadcast axes, sum) the gradient back to A and B.
        tensorA->get_grad()->add_to_data(offsetA, grad->data()[i]);
        tensorB->get_grad()->add_to_data(offsetB, grad->data()[i]);
    }
}

// Pads `strides` on the left with 0s up to `maxRank`, then zeroes out the
// stride of every axis where `shape` is 1 (broadcast axis), so that iterating
// along a broadcast axis always re-reads/re-writes the same element.
std::vector<size_t> MatSumOp::broadcast_strides(const std::vector<size_t> &strides, const std::vector<size_t> &shape, size_t maxRank) {
    std::vector<size_t> result = strides;
    result.insert(result.begin(), maxRank - result.size(), 0);

    for (size_t d = 0; d < maxRank; ++d) {
        if (shape[d] == 1)
            result[d] = 0;
    }

    return result;
}

// Given a flat output index `i` and the output's shape, decomposes `i` back
// into per-dimension coordinates (row-major, starting from the last
// dimension) and uses them, together with each tensor's broadcast strides,
// to compute the corresponding memory offset in A and B.
void MatSumOp::broadcast_offsets(size_t i, const std::vector<size_t> &outShape, const std::vector<size_t> &stridesA,
                                 const std::vector<size_t> &stridesB, size_t &offsetA, size_t &offsetB) {
    size_t maxRank = outShape.size();
    size_t temp = i;
    offsetA = 0;
    offsetB = 0;

    for (size_t d = 0; d < maxRank; ++d) {
        size_t coord = temp % outShape[maxRank - d - 1];
        temp = temp / outShape[maxRank - d - 1];

        offsetA += coord * stridesA[maxRank - d - 1];
        offsetB += coord * stridesB[maxRank - d - 1];
    }
}