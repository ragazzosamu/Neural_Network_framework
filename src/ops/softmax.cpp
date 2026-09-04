#include "ops/softmax.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Computes softmax(x) along the last dimension, applied independently to
// each "row" (broadcasting over every leading dimension): for an input of
// shape (..., row_length), each row of row_length elements is turned into a
// probability distribution that sums to 1.
std::shared_ptr<Tensor> SoftmaxOp::forward() {
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("The number of inputs must be 1");
    }

    const auto &input = o_inputs[0];

    // Guard against a null operand before dereferencing it below.
    if (input == nullptr) {
        throw std::invalid_argument("Input tensor must not be null");
    }

    auto data = input->data();
    std::vector<size_t> shape = input->shape();

    // shape.back() below requires at least one dimension.
    if (shape.empty()) {
        throw std::invalid_argument("Input tensor must have at least 1 dimension");
    }

    Tensor output(shape);

    // Softmax is applied to each row independently: in the common case the
    // input is a single row (a vector), but for a transformer it's typically
    // a matrix (or higher-rank tensor), where every row along the last
    // dimension gets its own softmax.
    size_t row_length = shape.back();

    // A zero-length row would make the division below undefined behaviour.
    if (row_length == 0) {
        throw std::invalid_argument("The last dimension must be non-zero");
    }

    size_t operation_number = input->size() / row_length;

    for (size_t i = 0; i < operation_number; ++i) {
        size_t offset = i * row_length;

        // Subtracting the row's max before exponentiating avoids overflow in
        // std::exp without changing the result (softmax is shift-invariant).
        float max_value = *std::max_element(&data[offset], &data[offset + row_length]);
        float sum = 0.0f;

        // Compute exp(x - max_value) for the row and accumulate the sum
        // needed to normalize it right after.
        for (size_t j = 0; j < row_length; ++j) {
            output.set_data(offset + j, std::exp(data[offset + j] - max_value));
            sum += output.data()[offset + j];
        }

        for (size_t j = 0; j < row_length; ++j) {
            output.set_data(offset + j, output.data()[offset + j] / sum);
        }
    }

    if (input->requires_grad()) {
        output.set_requires_grad(true);
        output.set_operation(shared_from_this());
    }

    // The output is needed again, unchanged, during backward() (see the
    // derivative formula there), so it's cached here.
    saved_output = std::make_shared<Tensor>(output.clone());

    return std::make_shared<Tensor>(std::move(output));
}

// Backward pass for y = softmax(x): for each row, using y (saved_output) and
// the incoming gradient g (grad), the derivative is
//   dx_j = y_j * (g_j - dot_product),  where dot_product = sum_k(g_k * y_k)
// (this follows from the softmax Jacobian dy_i/dx_j = y_i * (delta_ij - y_j)).
void SoftmaxOp::backward(std::shared_ptr<Tensor> grad) const {
    // Guard against a null gradient before dereferencing it below.
    if (grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }

    if (o_inputs.size() != 1) {
        throw std::invalid_argument("The number of inputs must be 1");
    }

    const auto &input = o_inputs[0];

    if (input == nullptr) {
        throw std::invalid_argument("Input tensor must not be null");
    }

    // forward() must have run at least once to populate saved_output, since
    // the derivative below is expressed in terms of y = softmax(x), not x.
    if (saved_output == nullptr) {
        throw std::invalid_argument("backward() called before forward(): no saved output available");
    }

    if (input->get_grad() == nullptr) {
        input->set_grad(std::make_shared<Tensor>(input->shape()));
    }

    std::vector<size_t> shape = input->shape();

    if (shape.empty()) {
        throw std::invalid_argument("Input tensor must have at least 1 dimension");
    }

    // Calculate the backward pass row-by-row, mirroring forward().
    size_t row_length = shape.back();

    if (row_length == 0) {
        throw std::invalid_argument("The last dimension must be non-zero");
    }

    size_t operation_number = input->size() / row_length;

    auto grad_input = input->get_grad();
    for (size_t i = 0; i < operation_number; ++i) {
        size_t offset = i * row_length;

        // dot_product = sum_k(g_k * y_k) for this row.
        float dot_product = 0.0f;
        for (size_t j = 0; j < row_length; ++j) {
            dot_product += grad->data()[offset + j] * saved_output->data()[offset + j];
        }

        for (size_t j = 0; j < row_length; ++j) {
            float dx_j = saved_output->data()[offset + j] * (grad->data()[offset + j] - dot_product);
            grad_input->add_to_data(offset + j, dx_j);
        }
    }
}