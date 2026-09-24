#include "ops/matadd.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

#ifdef NN_USE_OPENBLAS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <cblas.h>
#endif

// Computes C = A + B element-wise, with NumPy-style broadcasting on every
// dimension (unlike MatMulOp, there's no "special" last-two-dimensions
// treatment here: every axis is broadcast the same way).
std::shared_ptr<Tensor> MatAddOp::forward() {
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

    const float *dataA = tensorA->data();
    const float *dataB = tensorB->data();
    float *dataOutput = output.data();

    // Read once here (like the shapes above) instead of on every loop iteration.
    const std::vector<size_t> &tensorStridesA = tensorA->strides();
    const std::vector<size_t> &tensorStridesB = tensorB->strides();

    // The fast paths below walk the buffers with plain row-major indices
    // (i, or row * N + j), which is only valid for contiguous tensors: a
    // non-contiguous view (transpose()/permute()) must take the general path.
    const bool contiguous = is_contiguous(tensorA) && is_contiguous(tensorB);

    if (contiguous && shapeA == shapeB) {
        // Fast path: no broadcasting is actually needed (both operands already
        // have the exact same shape), so both tensors can be walked with a
        // single flat index, without going through the offset machinery below.
        for (size_t i = 0; i < output_size; ++i) {
            dataOutput[i] = dataA[i] + dataB[i];
        }
    } else if (contiguous && is_row_bias(tensorA->shape(), tensorB->shape())) {
        // Linear bias: B= [1, O] is added to each of the R rows of A = [R, O]. -> R = Batch_size
        // Z + b
        for (size_t i = 0; i < shapeA[0]; ++i) {

            size_t row_offset = i * tensorStridesA[0];
            for (size_t j = 0; j < shapeA[1]; ++j) {
                dataOutput[row_offset + j] = dataA[row_offset + j] + dataB[j];
            }
        }

    } else if (contiguous && is_channel_bias(tensorA->shape(), tensorB->shape())) {
        // Conv2D bias: A = [batch, C, P] (values), B = [C, 1] (bias).
        // Each (batch, channel) row of P elements gets the same constant
        // bias[channel]: the bias depends on the channel (i), not on the
        // position inside the row (j).

        for (size_t b = 0; b < shapeA[0]; ++b) {

            size_t batch_offset = b * tensorStridesA[0];
            for (size_t i = 0; i < shapeA[1]; ++i) {

                size_t row_offset = batch_offset + i * tensorStridesA[1];
                for (size_t j = 0; j < shapeA[2]; ++j) {

                    dataOutput[row_offset + j] = dataA[row_offset + j] + dataB[i];
                }
            }
        }

    } else {
        // General path: at least one axis is being broadcast, so A and B each
        // need their own (possibly repeated) offset for every output element.
        std::vector<size_t> stridesA = broadcast_strides(tensorStridesA, shapeA, maxRank);
        std::vector<size_t> stridesB = broadcast_strides(tensorStridesB, shapeB, maxRank);

        for (size_t i = 0; i < output_size; ++i) {
            size_t offsetA, offsetB;
            broadcast_offsets(i, shapeOutput, stridesA, stridesB, offsetA, offsetB);

            dataOutput[i] = dataA[offsetA] + dataB[offsetB];
        }
    }

    if (tensorA->requires_grad() || tensorB->requires_grad()) {
        output.set_requires_grad(true);
        output.set_operation(shared_from_this());
    }

    return std::make_shared<Tensor>(std::move(output));
}

// Backward pass for C = A + B: the local derivative of a sum with respect to
// each operand is 1, so the incoming gradient is simply routed to A and B
// unchanged, element by element — except that any axis A or B was broadcast
// over must be summed back over. Accumulating the same grad element into
// every broadcast position (via +=) is exactly what
// "collapses" that axis back down, which is the correct gradient for a
// broadcast sum.
void MatAddOp::backward(std::shared_ptr<Tensor> grad) const {
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

    const float *gradData = grad->data();
    float *gradA = tensorA->get_grad()->data();
    float *gradB = tensorB->get_grad()->data();
    const auto &gradShape = grad->shape();
    const size_t gradSize = grad->size();

    // Read once here (like the shapes above) instead of on every loop iteration.
    // In backward the buffers accessed are grad (read) and dA/dB (written):
    // A and B themselves are never read, so these are the strides that matter.
    const std::vector<size_t> &gradStrides = grad->strides();
    const std::vector<size_t> &gradStridesA = tensorA->get_grad()->strides();
    const std::vector<size_t> &gradStridesB = tensorB->get_grad()->strides();

    // Same layouts as in forward(): A = values, B = bias. The fast paths walk
    // grad, dA and dB with plain row-major indices, so those three buffers
    // must be contiguous (A and B don't need to be, since they are not read).
    const bool contiguous = is_contiguous(grad) && is_contiguous(tensorA->get_grad()) && is_contiguous(tensorB->get_grad());

    if (contiguous && tensorA->shape() == tensorB->shape()) {
        // Same shape: every element of A and B contributed to exactly one
        // output element, so both receive grad unchanged.
        for (size_t i = 0; i < gradSize; ++i) {
            gradA[i] += gradData[i];
            gradB[i] += gradData[i];
        }

    } else if (contiguous && is_row_bias(tensorA->shape(), tensorB->shape())) {
        // Linear bias: A = [R, N] (values), B = [1, N] (bias).
        //   dA = grad (same shape as the output).
        //   dB[column] = sum of grad over all the R rows of that column, since
        //   bias[column] was added to every row in forward().

        for (size_t i = 0; i < gradShape[0]; ++i) {

            size_t row_offset = i * gradStrides[0];
            for (size_t j = 0; j < gradShape[1]; ++j) {
                gradA[row_offset + j] += gradData[row_offset + j];
                gradB[j] += gradData[row_offset + j];
            }
        }

    } else if (contiguous && is_channel_bias(tensorA->shape(), tensorB->shape())) {
        // Conv2D bias: A = [batch, C, P] (values), B = [C, 1] (bias).
        //   dA = grad (same shape as the output).
        //   dB[channel] = sum of grad over every element of every
        //   (batch, channel) row with that channel: in forward() bias[channel]
        //   was added to all P positions of that row, in every batch, so its
        //   gradient collects all of those contributions.

        for (size_t b = 0; b < gradShape[0]; ++b) {

            size_t batch_offset = b * gradStrides[0];
            for (size_t i = 0; i < gradShape[1]; ++i) {

                size_t row_offset = batch_offset + i * gradStrides[1];
                float sum = 0.0f;
                for (size_t j = 0; j < gradShape[2]; ++j) {
                    gradA[row_offset + j] += gradData[row_offset + j];
                    sum += gradData[row_offset + j];
                }
                gradB[i] += sum;
            }
        }

    } else {
        // General path: at least one axis is being broadcast in an arbitrary way.
        size_t maxRank = std::max(shapeA.size(), shapeB.size());

        shapeA.insert(shapeA.begin(), maxRank - shapeA.size(), 1);
        shapeB.insert(shapeB.begin(), maxRank - shapeB.size(), 1);

        // Broadcast-adjusted strides: on a broadcast axis (size 1), the stride is
        // zeroed, so every position along that axis maps back to the same element
        // of A/B — this is what makes the += below accumulate (sum) the
        // gradient over that axis instead of overwriting it.
        //
        // The strides come from the gradient tensors (not from A and B): dA and dB
        // have their own (normally contiguous) layout, which differs from A's/B's
        // when A or B is a non-contiguous view.
        std::vector<size_t> stridesA = broadcast_strides(gradStridesA, shapeA, maxRank);
        std::vector<size_t> stridesB = broadcast_strides(gradStridesB, shapeB, maxRank);

        for (size_t i = 0; i < gradSize; ++i) {
            size_t offsetA, offsetB;
            broadcast_offsets(i, gradShape, stridesA, stridesB, offsetA, offsetB);

            // Route (and, on broadcast axes, sum) the gradient back to A and B.
            gradA[offsetA] += gradData[i];
            gradB[offsetB] += gradData[i];
        }
    }
}

// ---------- Private Helper ----------

// A = [R, N] (values), B = [1, N] (bias).
bool MatAddOp::is_row_bias(const std::vector<size_t> &shapeA, const std::vector<size_t> &shapeB) {
    return shapeA.size() == 2 && shapeB.size() == 2 && shapeB[0] == 1 && shapeB[1] == shapeA[1];
}

// A = [batch, C, P] (values), B = [C, 1] (bias).
bool MatAddOp::is_channel_bias(const std::vector<size_t> &shapeA, const std::vector<size_t> &shapeB) {
    return shapeA.size() == 3 && shapeB.size() == 2 && shapeB[1] == 1 && shapeB[0] == shapeA[1];
}

// Pads `strides` on the left with 0s up to `maxRank`, then zeroes out the
// stride of every axis where `shape` is 1 (broadcast axis), so that iterating
// along a broadcast axis always re-reads/re-writes the same element.
std::vector<size_t> MatAddOp::broadcast_strides(const std::vector<size_t> &strides, const std::vector<size_t> &shape, size_t maxRank) {
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
void MatAddOp::broadcast_offsets(size_t i, const std::vector<size_t> &outShape, const std::vector<size_t> &stridesA,
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
