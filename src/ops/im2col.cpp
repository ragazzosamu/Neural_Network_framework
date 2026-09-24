#include "ops/im2col.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
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
    const float *input_data = input->data();
    const float *grad_data = is_backward ? grad->data() : nullptr;
    float *input_grad = nullptr;
    float *output_data = nullptr;

    if (is_backward) {
        if (input->get_grad() == nullptr) {
            input->set_grad(std::make_shared<Tensor>(input_shape));
        }
        input_grad = input->get_grad()->data();
    } else {
        output = std::make_shared<Tensor>(std::vector<size_t>{batch_size, values_per_patch, patches_number});
        output_data = output->data();
    }

    // The image-side index is computed with the strides of the tensor that is
    // actually accessed: the input (read) in forward, its gradient (written)
    // in backward. They differ when the input is a non-contiguous view, since
    // the gradient is always allocated with its own contiguous layout.
    const auto &input_strides = is_backward ? input->get_grad()->strides() : input->strides();

    // Signed copies of the sizes used in the valid-range formulas below: those
    // formulas involve negative offsets, and mixing them with size_t would turn
    // the arithmetic unsigned.
    const std::int64_t S = static_cast<std::int64_t>(kernel_stride);
    const std::int64_t W = static_cast<std::int64_t>(input_width);
    const std::int64_t H = static_cast<std::int64_t>(input_height);
    const std::int64_t OW = static_cast<std::int64_t>(output_width);

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
                    const size_t output_row_index = (channel * kernel_height + kernel_y) * kernel_width + kernel_x;

                    // For a fixed kernel element, the input position seen from each output
                    // position is:
                    //   input_y = output_y * stride + off_y
                    //   input_x = output_x * stride + off_x
                    // The offsets depend only on (kernel_y, kernel_x), so they are computed
                    // once here instead of once per output element. They are signed
                    // because padding can make them negative.
                    const std::int64_t off_y = static_cast<std::int64_t>(kernel_y * dilatation) - static_cast<std::int64_t>(padding);
                    const std::int64_t off_x = static_cast<std::int64_t>(kernel_x * dilatation) - static_cast<std::int64_t>(padding);

                    // Valid range [start, end) of output_x: the positions whose
                    // input_x falls inside the image (0 <= input_x <= W - 1). Everything
                    // before start and from end on reads the zero padding.

                    // Lower bound: input_x >= 0  ->  output_x >= -off_x / S.
                    // start is the FIRST output_x satisfying it, so the division is
                    // rounded up.
                    // If off_x >= 0, even output_x = 0 is already inside the image.
                    std::int64_t start = off_x >= 0 ? 0 : (-off_x + S - 1) / S;

                    // Upper bound: input_x <= W - 1  ->  output_x <= (W - 1 - off_x) / S.
                    // The LAST valid output_x is wanted, so the integer division (rounding
                    // down) is correct; + 1 turns it into the exclusive end used by
                    // `output_x < end`. The `< 0` case (empty range) is handled explicitly
                    // because integer division rounds a negative result towards zero.
                    std::int64_t end = (W - 1 - off_x) < 0 ? 0 : (W - 1 - off_x) / S + 1;
                    // The formulas only look at the image edges, not at the length of the
                    // output row: clamp end to it, and keep start <= end so an
                    // empty range stays empty.
                    end = std::min(end, OW);
                    start = std::min(start, end);

                    for (size_t output_y = 0; output_y < output_height; ++output_y) {
                        const std::int64_t input_y = static_cast<std::int64_t>(output_y) * S + off_y;
                        const size_t row_output_offset =
                            batch * values_per_patch * patches_number + output_row_index * patches_number + output_y * output_width;

                        // Whole segment in the top/bottom padding: the output is already zero
                        // and the padding receives no gradient.
                        if (input_y < 0 || input_y >= H) {
                            continue;
                        }

                        const size_t row_input_offset = channel_offset + static_cast<size_t>(input_y) * input_strides[2];

                        if (is_backward) {
                            for (std::int64_t output_x = start; output_x < end; ++output_x) {
                                input_grad[row_input_offset + (output_x * S + off_x) * input_strides[3]] += grad_data[row_output_offset + output_x];
                            }
                        } else if (S == 1 && input_strides[3] == 1) {
                            // For a fixed kernel element, consecutive patches along a row
                            // (output_x, output_x + 1, ...) see image pixels that are `stride`
                            // columns apart. With stride 1 (and a contiguous image row) those
                            // pixels are consecutive in memory, just like their destination
                            // in the column matrix, so the whole valid part [start, end) is a
                            // single block copy: output_x = start maps to input_x = start + off_x.
                            std::memcpy(output_data + row_output_offset + start, input_data + row_input_offset + start + off_x,
                                        static_cast<size_t>(end - start) * sizeof(float)); // output_x = start -> input_x = start + off_x
                        } else {
                            // Stride > 1: elements are not consecutive, but there is no padding check.
                            for (std::int64_t output_x = start; output_x < end; ++output_x) {
                                output_data[row_output_offset + output_x] = input_data[row_input_offset + (output_x * S + off_x) * input_strides[3]];
                            }
                        }
                    }
                }
            }
        }
    }

    return output;
}
