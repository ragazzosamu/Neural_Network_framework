#include "nn/conv2D.hpp"
#include "ops/matadd.hpp"

Conv2D::Conv2D(size_t C_in, size_t C_out, size_t kernel_height, size_t kernel_width, size_t padding, size_t stride, size_t dilatation)
    : kernel_height(kernel_height), kernel_width(kernel_width), padding(padding), stride(stride), dilatation(dilatation) {

    std::vector<size_t> shape_w = {C_out, C_in, kernel_height, kernel_width};
    std::shared_ptr<Tensor> weights = std::make_shared<Tensor>(shape_w);

    std::normal_distribution<float> dis(0.0f, 1.0f);
    auto &gen = rng::engine();

    float *weights_data = weights->data();
    for (size_t i = 0; i < weights->size(); ++i) {
        weights_data[i] = dis(gen);
    }

    std::vector<size_t> shape_b = {C_out, 1};
    std::shared_ptr<Tensor> biases = std::make_shared<Tensor>(shape_b);

    float *biases_data = biases->data();
    for (size_t i = 0; i < biases->size(); ++i) {
        biases_data[i] = 0.0f;
    }

    params = {weights, biases};
}

std::shared_ptr<Tensor> Conv2D::forward(const std::shared_ptr<Tensor> &input) const {
    try {
        if (input == nullptr) {
            throw std::invalid_argument("Input tensor must not be null");
        }

        auto im2col_op =
            std::make_shared<im2ColOp>(std::vector<std::shared_ptr<Tensor>>{input}, kernel_height, kernel_width, padding, stride, dilatation);
        std::shared_ptr<Tensor> col_input = im2col_op->forward();

        auto weight2col_op = std::make_shared<Weight2ColOp>(std::vector<std::shared_ptr<Tensor>>{params[0]});
        std::shared_ptr<Tensor> col_weights = weight2col_op->forward();

        std::vector<std::shared_ptr<Tensor>> multiplication_params{col_weights, col_input};
        std::shared_ptr<MatMulOp> multiplication = std::make_shared<MatMulOp>(multiplication_params);
        std::shared_ptr<Tensor> mul_output = multiplication->forward();

        auto sum_params = std::vector<std::shared_ptr<Tensor>>{mul_output, params[1]};
        std::shared_ptr<MatAddOp> sum = std::make_shared<MatAddOp>(sum_params);
        std::shared_ptr<Tensor> sum_output = sum->forward();

        const auto &input_shape = input->shape();
        const size_t input_height = input_shape[2];
        const size_t input_width = input_shape[3];
        const size_t effective_kernel_height = dilatation * (kernel_height - 1) + 1;
        const size_t effective_kernel_width = dilatation * (kernel_width - 1) + 1;
        const size_t output_height = (input_height + 2 * padding - effective_kernel_height) / stride + 1;
        const size_t output_width = (input_width + 2 * padding - effective_kernel_width) / stride + 1;

        auto col2im_op = std::make_shared<col2ImOp>(std::vector<std::shared_ptr<Tensor>>{sum_output}, output_height, output_width);
        std::shared_ptr<Tensor> img_output = col2im_op->forward();

        return img_output;
    } catch (const std::exception &e) {
        std::throw_with_nested(std::runtime_error(std::string("Conv2D error: ") + e.what()));
    }
}