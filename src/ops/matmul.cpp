#include "ops/matmul.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

// Computes C = A @ B with NumPy-style broadcasting on all dimensions
// except the last two (which are treated as the actual matrix rows/columns).
std::shared_ptr<Tensor> MatMulOp::forward() {
    if (o_inputs.size() != 2) {
        throw std::invalid_argument("The number of inputs must be 2");
    }

    const auto &tensorA = o_inputs[0];
    const auto &tensorB = o_inputs[1];

    if (tensorA == nullptr || tensorB == nullptr) {
        throw std::invalid_argument("Input tensors must not be null");
    }

    std::vector<size_t> shapeA = tensorA->shape();
    std::vector<size_t> shapeB = tensorB->shape();

    size_t maxRank = std::max(shapeA.size(), shapeB.size());

    // Matrix multiplication needs at least a row and a column dimension on each
    // side. Without this check, maxRank < 2 would make `maxRank - 2` underflow
    // (size_t is unsigned) and cause out-of-bounds accesses further down.
    if (maxRank < 2) {
        throw std::invalid_argument("Both tensors must have at least 2 dimensions to be multiplied");
    }

    // Pad the shorter shape with leading 1s so both tensors have the same rank
    // (this is what makes the broadcasting rules below simple to apply).
    shapeA.insert(shapeA.begin(), maxRank - shapeA.size(), 1);
    shapeB.insert(shapeB.begin(), maxRank - shapeB.size(), 1);

    std::vector<size_t> shapeOutput(maxRank);
    if (shapeA[maxRank - 1] != shapeB[maxRank - 2]) {
        throw std::invalid_argument("The two shapes are not compatible for moltiplication");
    }
    // Last two dims follow standard matrix-multiplication rules: (n x m) @ (m x p) = (n x p).
    shapeOutput[maxRank - 1] = shapeB[maxRank - 1];
    shapeOutput[maxRank - 2] = shapeA[maxRank - 2];

    // All the other (batch) dimensions follow standard broadcasting: equal, or one of them is 1.
    for (size_t i = 0; i < maxRank - 2; ++i) {
        if (shapeA[i] != shapeB[i] && shapeA[i] != 1 && shapeB[i] != 1) {
            throw std::invalid_argument("The two shapes are not compatible for broadcasting");
        }
        shapeOutput[i] = std::max(shapeA[i], shapeB[i]);
    }

    Tensor output(shapeOutput);
    size_t output_size = output.size();

    // Note: forcing a stride to 0 makes every index along that axis map back to the same
    // (first) element in memory. That trick was used by the original ("sol1") approach
    // below to compute the dot product; it is not needed by the current implementation.

    // Broadcast-adjusted strides (batch dimensions with size 1 get stride 0, so that
    // iterating over them always points back to the same, single, slice of the tensor).
    std::vector<size_t> stridesA = broadcast_strides(tensorA->strides(), shapeA, maxRank);
    std::vector<size_t> stridesB = broadcast_strides(tensorB->strides(), shapeB, maxRank);
    std::vector<size_t> stridesOutput = broadcast_strides(output.strides(), shapeOutput, maxRank);

    // Approach 2 ("sol2")

    // Original (non-broadcast) strides: needed to know the real step size in memory
    // when moving along the row/column of the actual matrices (not the broadcast view).
    size_t A_column_number = shapeA[maxRank - 1];
    size_t A_row_number = shapeA[maxRank - 2];
    size_t B_column_number = shapeB[maxRank - 1];

    size_t stride_column_A = stridesA[maxRank - 1];
    size_t stride_row_A = stridesA[maxRank - 2];

    size_t stride_column_B = stridesB[maxRank - 1];
    size_t stride_row_B = stridesB[maxRank - 2];

    size_t stride_column_output = stridesOutput[maxRank - 1];
    size_t stride_row_output = stridesOutput[maxRank - 2];

    // A zero-sized row/column dimension would make the division below
    // (0 / 0, or output_size / 0) undefined behaviour instead of just an empty result.
    if (A_row_number == 0 || B_column_number == 0) {
        throw std::invalid_argument("Matrix dimensions must be non-zero");
    }

    // Total number of matrices (batches) to multiply.
    size_t iteration_number = output_size / (A_row_number * B_column_number);

    // Cache the underlying buffers/shape once, instead of re-invoking these accessors
    // on every single access inside the loops below.
    const auto &dataA = tensorA->data();
    const auto &dataB = tensorB->data();
    const auto &outShape = output.shape();

    for (size_t b = 0; b < iteration_number; ++b) {

        size_t offsetA, offsetB, offsetOutput;
        broadcast_offsets(b, outShape, stridesA, stridesB, stridesOutput, offsetA, offsetB, offsetOutput);
        // offsetA/offsetB/offsetOutput now point to the first element of the
        // current batch's matrices.

        for (size_t i = 0; i < A_row_number; ++i) {
            // Row-only parts of the offsets, hoisted out of the k loop since they
            // don't depend on k.
            size_t row_offset_A = offsetA + i * stride_row_A;
            size_t row_offset_output = offsetOutput + i * stride_row_output;

            for (size_t k = 0; k < A_column_number; ++k) {
                float valA = dataA[row_offset_A + k * stride_column_A];
                // valA is loaded once and reused for every j below, since this single
                // A element contributes to every element of output row i for this k.

                // Column-only part of B's offset (w.r.t. k), hoisted out of the j loop.
                size_t col_offset_B = offsetB + k * stride_row_B;

                for (size_t j = 0; j < B_column_number; ++j) {
                    float value = valA * dataB[col_offset_B + j * stride_column_B];
                    size_t output_index = row_offset_output + j * stride_column_output;
                    output.add_to_data(output_index, value);
                }
            }
        }
    }

    if (tensorA->requires_grad() || tensorB->requires_grad()) {
        output.set_requires_grad(true);
        output.set_operation(shared_from_this());
    }

    return std::make_shared<Tensor>(std::move(output));
}

// Approach 1 ("sol1"), deprecated: functionally correct, but too slow.
/*
stridesA[maxRank - 1] = 0;
stridesB[maxRank - 2] = 0;

for (size_t i = 0; i < output_size; ++i) {
    size_t offsetA, offsetB;
    broadcast_offsets(i, shapeOutput, stridesA, stridesB, offsetA, offsetB);

    size_t operation_numbers = shapeA[maxRank - 1];
    float value = 0;

    // Deprecated: fewer nested loops, but memory accesses jump around non-contiguously,
    // which hurts cache locality.
    for (size_t j = 0; j < operation_numbers; ++j) {
        // Use the original column stride of A and row stride of B to compute the dot product.
        value += (offsetA + j * tensorA->strides()[maxRank - 1]) *
                 (offsetB + j * tensorB->strides()[maxRank - 2]);
    }

    output.set_data(i, value);
}
*/

// Backward pass for C = A @ B: given dC (grad), accumulates dA and dB into
// A's and B's gradient tensors.
void MatMulOp::backward(std::shared_ptr<Tensor> grad) const {

    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }

    if (o_inputs.size() != 2) {
        throw std::invalid_argument("The number of inputs must be 2");
    }

    const auto &tensorA = o_inputs[0];
    const auto &tensorB = o_inputs[1];

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

    // Same reasoning as in forward(): with maxRank < 2, `maxRank - 2` would
    // underflow (size_t is unsigned) and cause out-of-bounds accesses below.
    if (maxRank < 2) {
        throw std::invalid_argument("Both tensors must have at least 2 dimensions");
    }

    shapeA.insert(shapeA.begin(), maxRank - shapeA.size(), 1);
    shapeB.insert(shapeB.begin(), maxRank - shapeB.size(), 1);

    // Broadcast-adjusted strides, used to find each batch's offset.
    std::vector<size_t> stridesA = broadcast_strides(tensorA->strides(), shapeA, maxRank);
    std::vector<size_t> stridesB = broadcast_strides(tensorB->strides(), shapeB, maxRank);
    std::vector<size_t> stridesgrad = broadcast_strides(grad->strides(), grad->shape(), maxRank);

    // Original (non-broadcast) strides: the real step size in memory for each matrix.
    size_t A_column_number = shapeA[maxRank - 1];
    size_t B_row_number = shapeB[maxRank - 2];
    size_t grad_column_number = grad->shape()[maxRank - 1];
    size_t grad_row_number = grad->shape()[maxRank - 2];

    size_t stride_column_A = tensorA->strides()[maxRank - 1];
    size_t stride_row_A = tensorA->strides()[maxRank - 2];

    size_t stride_column_B = tensorB->strides()[maxRank - 1];
    size_t stride_row_B = tensorB->strides()[maxRank - 2];

    size_t stride_column_grad = grad->strides()[maxRank - 1];
    size_t stride_row_grad = grad->strides()[maxRank - 2];

    // A zero-sized row/column dimension would make the division below
    // undefined behaviour instead of just an empty result.
    if (grad_row_number == 0 || grad_column_number == 0) {
        throw std::invalid_argument("Gradient matrix dimensions must be non-zero");
    }

    // Total number of matrices (batches).
    size_t iteration_number = grad->size() / (grad_row_number * grad_column_number);

    // Cache the underlying buffers/shape/gradient-tensors once, instead of
    // re-invoking these accessors on every single access inside the loops below.
    const auto &dataA = tensorA->data();
    const auto &dataB = tensorB->data();
    const auto &dataGrad = grad->data();
    const auto &gradShape = grad->shape();
    const auto &gradA = tensorA->get_grad();
    const auto &gradB = tensorB->get_grad();

    for (size_t b = 0; b < iteration_number; ++b) {
        size_t offsetA, offsetB, offsetgrad;
        broadcast_offsets(b, gradShape, stridesA, stridesB, stridesgrad, offsetA, offsetB, offsetgrad);

        for (size_t i = 0; i < grad_row_number; ++i) {
            // Row-only parts of the offsets, hoisted out of the k loop since they
            // don't depend on k.
            size_t row_offset_A = offsetA + i * stride_row_A;
            size_t row_offset_grad = offsetgrad + i * stride_row_grad;

            for (size_t j = 0; j < B_row_number; ++j) {
                size_t A_idx = row_offset_A + j * stride_column_A;
                float valA = dataA[A_idx];
                size_t row_offset_B = offsetB + j * stride_row_B;

                float acc = 0.0f;
                // For this row i of the gradient, loop over each row j of B (equivalently, each
                // column of B^T). For every j, scanning along k both scatters the corresponding
                // contribution into dB, one element at a time, and simultaneously accumulates
                // the running dot-product sum that becomes dA[i,j].
                for (size_t k = 0; k < grad_column_number; ++k) {
                    float valgrad = dataGrad[row_offset_grad + k * stride_column_grad];
                    size_t B_idx = row_offset_B + k * stride_column_B;

                    gradB->add_to_data(B_idx, valA * valgrad);
                    acc += valgrad * dataB[B_idx];
                }

                gradA->add_to_data(A_idx, acc);
            }
        }
    }
}

// ---------- Private Helper ----------

// Given a flat batch index `i` (0..number_of_batches-1) and the output's batch shape,
// decomposes `i` back into per-dimension coordinates (row-major, starting from the
// last batch dimension) and uses them, together with each tensor's broadcast strides,
// to compute the memory offset of that batch's matrix in A, B and the output.
void MatMulOp::broadcast_offsets(size_t i, const std::vector<size_t> &outShape, const std::vector<size_t> &stridesA,
                                 const std::vector<size_t> &stridesB, const std::vector<size_t> &stridesOutput, size_t &offsetA, size_t &offsetB,
                                 size_t &offsetOutput) {
    size_t maxRank = outShape.size();
    size_t temp = i;
    offsetA = 0;
    offsetB = 0;
    offsetOutput = 0;

    // Only the batch dimensions (everything except the last two, which are the
    // actual matrix rows/columns) are decomposed here.
    for (size_t d = 0; d < maxRank - 2; ++d) {
        size_t coord = temp % outShape[maxRank - d - 3];
        temp = temp / outShape[maxRank - d - 3];

        offsetA += coord * stridesA[maxRank - d - 3];
        offsetB += coord * stridesB[maxRank - d - 3];
        offsetOutput += coord * stridesOutput[maxRank - d - 3];
    }
}

// Pads `strides` on the left with 0s up to `maxRank`, then zeroes out the stride of
// every axis where `shape` is 1 (broadcast axis), so that iterating along a broadcast
// axis always re-reads the same element.
std::vector<size_t> MatMulOp::broadcast_strides(const std::vector<size_t> &strides, const std::vector<size_t> &shape, size_t maxRank) {
    std::vector<size_t> result = strides;
    result.insert(result.begin(), maxRank - result.size(), 0);

    for (size_t d = 0; d < maxRank; ++d) {
        if (shape[d] == 1)
            result[d] = 0;
    }

    return result;
}