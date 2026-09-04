#include "nn/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/matsum.hpp"

#include <random>

Linear::Linear(size_t input_size, size_t output_size) {

    // Weight matrix has input_size rows and output_size columns: each of the
    // output_size outputs is a combination of all input_size inputs, so each
    // output "owns" a column of input_size weights.
    std::vector<size_t> shape_w = {input_size, output_size};
    std::shared_ptr<Tensor> weights = std::make_shared<Tensor>(shape_w);

    // Random initialization: sampling from a standard normal distribution
    // avoids the symmetry issues of a constant initial weight (all neurons
    // would otherwise learn identical features).
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1);

    for (size_t i = 0; i < weights->size(); ++i) {
        weights->set_data(i, dis(gen));
    }

    // Bias is one value per output feature, initialized to zero. Starting
    // at zero is safe for biases (unlike weights), since the bias is added after the weighted sum of inputs,
    // so it doesn't affect the symmetry of the initial feature extraction.
    std::vector<size_t> shape_b = {1, output_size};
    std::shared_ptr<Tensor> biases = std::make_shared<Tensor>(shape_b);

    for (size_t i = 0; i < biases->size(); ++i) {
        biases->set_data(i, 0.0f);
    }

    params = {weights, biases};
}

// Computes output = input * W + bias.
//
// This follows the common convention where the input is treated as a row
// vector, so the layer computes x * W + b rather than W * x + b — this is
// why weights are shaped {input_size, output_size} rather than
// {output_size, input_size}.
//
// Any exception raised inside MatMulOp/MatSumOp (e.g. a shape mismatch, or
// a null input reaching the multiplication) is caught and re-thrown as a
// std::runtime_error prefixed with "Linear error: ". This gives callers a
// single, predictable exception type to catch regardless of which internal
// operation failed, while std::throw_with_nested keeps the original
// exception attached (recoverable via std::rethrow_if_nested) for anyone
// who needs the precise underlying cause.
std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor> &input) const {
    try {
        if (input == nullptr) {
            throw std::invalid_argument("Input tensor must not be null");
        }

        auto multiplication_params = {input, params[0]};
        std::shared_ptr<MatMulOp> multiplication = std::make_shared<MatMulOp>(multiplication_params);
        std::shared_ptr<Tensor> mul_output = multiplication->forward();

        auto sum_params = {params[1], mul_output};
        std::shared_ptr<MatSumOp> sum = std::make_shared<MatSumOp>(sum_params);
        std::shared_ptr<Tensor> sum_output = sum->forward();

        return sum_output;
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error(std::string("Linear error: ") + e.what()));
    }
}