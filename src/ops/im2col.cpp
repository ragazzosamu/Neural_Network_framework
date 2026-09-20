#include "ops/im2col.hpp"
#include <cstdint>
#include <stdexcept>

im2ColOp::im2ColOp(std::vector<std::shared_ptr<Tensor>> inputs, size_t kernel_height, size_t kernel_width, size_t padding, size_t kernel_stride,
                   size_t dilatation)
    : Operation(std::move(inputs)), kernel_height(kernel_height), kernel_width(kernel_width), padding(padding), kernel_stride(kernel_stride),
      dilatation(dilatation) {}

std::shared_ptr<Tensor> im2ColOp::forward() {
    auto output = process(false, nullptr);

    if (o_inputs[0]->requires_grad()) {
        output->set_requires_grad(true);
        output->set_operation(shared_from_this());
    }
    return output;
}

void im2ColOp::backward(std::shared_ptr<Tensor> grad) const { process(true, grad); }

std::shared_ptr<Tensor> im2ColOp::process(bool is_backward, const std::shared_ptr<Tensor> &grad) const {
    if (o_inputs.size() != 1) {
        throw std::invalid_argument("Im2ColOp expects exactly one input");
    }
    if (is_backward && grad == nullptr) {
        throw std::invalid_argument("Gradient tensor must not be null");
    }
    if (kernel_height == 0 || kernel_width == 0 || kernel_stride == 0 || dilatation == 0) {
        throw std::invalid_argument("Kernel dimensions, stride and dilation must be non-zero");
    }

    const auto &input = o_inputs[0];
    if (input == nullptr) {
        throw std::invalid_argument("Input tensor must not be null");
    }

    const auto &input_shape = input->shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Im2ColOp expects an input with shape [B, C, H, W]");
    }

    const auto &input_strides = input->strides();
    const size_t batch_size = input_shape[0];
    const size_t channels = input_shape[1];
    const size_t input_height = input_shape[2];
    const size_t input_width = input_shape[3];
    const size_t effective_kernel_height = dilatation * (kernel_height - 1) + 1;
    const size_t effective_kernel_width = dilatation * (kernel_width - 1) + 1;
    const size_t padded_height = input_height + 2 * padding;
    const size_t padded_width = input_width + 2 * padding;

    if (padded_height < effective_kernel_height || padded_width < effective_kernel_width) {
        throw std::invalid_argument("Effective kernel is larger than the padded input");
    }

    const size_t output_height = (padded_height - effective_kernel_height) / kernel_stride + 1;
    const size_t output_width = (padded_width - effective_kernel_width) / kernel_stride + 1;
    const size_t patches_number = output_height * output_width;
    const size_t values_per_patch = channels * kernel_height * kernel_width;

    if (is_backward && grad->shape() != std::vector<size_t>{batch_size, values_per_patch, patches_number}) {
        throw std::invalid_argument("Gradient shape does not match Im2Col output");
    }

    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> input_grad;
    const auto input_data = input->data();
    const auto grad_data = is_backward ? grad->data() : nullptr;

    if (is_backward) {
        if (input->get_grad() == nullptr) {
            input->set_grad(std::make_shared<Tensor>(input_shape));
        }
        input_grad = input->get_grad();
    } else {
        output = std::make_shared<Tensor>(std::vector<size_t>{batch_size, values_per_patch, patches_number});
    }

    for (size_t batch = 0; batch < batch_size; ++batch) {
        const size_t batch_offset = batch * input_strides[0];

        // The same loop is used in both directions. Its order is:
        // channel -> kernel_y -> kernel_x -> output_y -> output_x.
        // It fixes one relative kernel value, visits every patch, and completes
        // one row of [kernel_value, patch] at a time.
        //
        // The alternative order
        // output_y -> output_x -> channel -> kernel_y -> kernel_x
        // copies one complete patch at a time. It produces the same matrix, but
        // this order makes patch_index consecutive within one matrix row.
        for (size_t channel = 0; channel < channels; ++channel) {
            const size_t channel_offset = batch_offset + channel * input_strides[1];

            for (size_t kernel_y = 0; kernel_y < kernel_height; ++kernel_y) {
                for (size_t kernel_x = 0; kernel_x < kernel_width; ++kernel_x) {
                    const size_t row_index = (channel * kernel_height + kernel_y) * kernel_width + kernel_x;

                    for (size_t output_y = 0; output_y < output_height; ++output_y) {
                        const std::int64_t input_y = static_cast<std::int64_t>(output_y * kernel_stride) - static_cast<std::int64_t>(padding) +
                                                     static_cast<std::int64_t>(kernel_y * dilatation);

                        for (size_t output_x = 0; output_x < output_width; ++output_x) {
                            const std::int64_t input_x = static_cast<std::int64_t>(output_x * kernel_stride) - static_cast<std::int64_t>(padding) +
                                                         static_cast<std::int64_t>(kernel_x * dilatation);
                            const size_t patch_index = output_y * output_width + output_x;
                            const size_t matrix_index = batch * values_per_patch * patches_number + row_index * patches_number + patch_index;

                            const bool is_padding = input_y < 0 || input_y >= static_cast<std::int64_t>(input_height) || input_x < 0 ||
                                                    input_x >= static_cast<std::int64_t>(input_width);

                            if (is_backward) {
                                if (!is_padding) {
                                    const size_t input_index = channel_offset + static_cast<size_t>(input_y) * input_strides[2] +
                                                               static_cast<size_t>(input_x) * input_strides[3];
                                    input_grad->add_to_data(input_index, grad_data[matrix_index]);
                                }
                            } else if (is_padding) {
                                output->set_data(matrix_index, 0.0f);
                            } else {
                                const size_t input_index = channel_offset + static_cast<size_t>(input_y) * input_strides[2] +
                                                           static_cast<size_t>(input_x) * input_strides[3];
                                output->set_data(matrix_index, input_data[input_index]);
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}
