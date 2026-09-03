#include "nn/linear.hpp"
#include "ops/matmul.hpp"
#include "ops/matsum.hpp"

Linear::Linear(size_t input_size, size_t output_size) {

    // for the weight i have output_size rows and as columns the input_size. So for each output i have input_size weights ( voglio che questo commento
    // lo tieni magari riadattandolo)
    std::vector<size_t> shape_w = {output_size, input_size};
    std::shared_ptr<Tensor> weights = std::make_shared<Tensor>(shape_w);

    std::vector<size_t> shape_b = {output_size, 1};
    std::shared_ptr<Tensor> biases = std::make_shared<Tensor>(shape_b);

    params = {weights, biases};
}

// W * x + B
std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor> &input) const {

    auto multiplication_params = {params[0], input};
    std::shared_ptr<MatMulOp> multiplication = std::make_shared<MatMulOp>(multiplication_params);
    std::shared_ptr<Tensor> mul_output = multiplication->forward();

    auto sum_params = {params[0], mul_output};
    std::shared_ptr<MatSumOp> sum = std::make_shared<MatSumOp>(sum_params);
    std::shared_ptr<Tensor> sum_output = sum->forward();

    return sum_output;
}