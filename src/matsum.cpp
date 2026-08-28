#include "matsum.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

Tensor MatSumOp::forward() {
    if (o_inputs.size() != 2) {
        throw std::invalid_argument("The number of inputs must be 2");
    }

    const auto &tensorA = o_inputs[0];
    std::vector<size_t> shapeA = tensorA->shape();

    const auto &tensorB = o_inputs[1];
    std::vector<size_t> shapeB = tensorB->shape();

    size_t maxRank = std::max(shapeA.size(), shapeB.size());

    shapeA.insert(shapeA.begin(), maxRank - shapeA.size(), 1);
    shapeB.insert(shapeB.begin(), maxRank - shapeB.size(), 1);

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
        for (size_t i = 0; i < output_size; ++i) {
            output.set_data(i, tensorA->data()[i] + tensorB->data()[i]);
        }
    } else {
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

void MatSumOp::backward(std::shared_ptr<Tensor> grad) const {
    const auto &tensorA = o_inputs[0];
    std::vector<size_t> shapeA = tensorA->shape();
    if (tensorA->get_grad() == nullptr) {
        tensorA->set_grad(std::make_shared<Tensor>(shapeA));
    }

    const auto &tensorB = o_inputs[1];
    std::vector<size_t> shapeB = tensorB->shape();
    if (tensorB->get_grad() == nullptr) {
        tensorB->set_grad(std::make_shared<Tensor>(shapeB));
    }

    size_t maxRank = std::max(shapeA.size(), shapeB.size());

    shapeA.insert(shapeA.begin(), maxRank - shapeA.size(), 1);
    shapeB.insert(shapeB.begin(), maxRank - shapeB.size(), 1);

    std::vector<size_t> stridesA = broadcast_strides(tensorA->strides(), shapeA, maxRank);
    std::vector<size_t> stridesB = broadcast_strides(tensorB->strides(), shapeB, maxRank);

    for (size_t i = 0; i < grad->size(); ++i) {
        size_t offsetA, offsetB;
        broadcast_offsets(i, grad->shape(), stridesA, stridesB, offsetA, offsetB);

        tensorA->get_grad()->add_to_data(offsetA, grad->data()[i]);
        tensorB->get_grad()->add_to_data(offsetB, grad->data()[i]);
    }
}

std::vector<size_t> MatSumOp::broadcast_strides(const std::vector<size_t> &strides,
                                                const std::vector<size_t> &shape, size_t maxRank) {
    std::vector<size_t> result = strides;
    result.insert(result.begin(), maxRank - result.size(), 0);

    for (size_t d = 0; d < maxRank; ++d) {
        if (shape[d] == 1)
            result[d] = 0;
    }

    return result;
}

void MatSumOp::broadcast_offsets(size_t i, const std::vector<size_t> &outShape,
                                 const std::vector<size_t> &stridesA,
                                 const std::vector<size_t> &stridesB, size_t &offsetA,
                                 size_t &offsetB) {
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