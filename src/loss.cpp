
#include "loss.hpp"
#include <stdexcept>
#include <string>

void Loss::validate_inputs(std::shared_ptr<Tensor> &model_output, std::shared_ptr<Tensor> &target) {
    if (!model_output || !target) {
        throw std::invalid_argument("Loss: model output and target tensors must not be null");
    }
    // Shape equality (not just matching total size) mirrors the check
    // CrossEntropyOp::forward() does internally, and also catches cases
    // like {2,3} vs {3,2} that have the same element count but are not
    // actually compatible.
    if (model_output->shape() != target->shape()) {
        throw std::invalid_argument("Loss: model output and target shape mismatch (sizes " + std::to_string(model_output->size()) + " vs " +
                                    std::to_string(target->size()) + ")");
    }
}

std::shared_ptr<Tensor> Loss::cross_entropy(std::shared_ptr<Tensor> &model_output, std::shared_ptr<Tensor> &target) {
    validate_inputs(model_output, target);

    std::shared_ptr<Tensor> softmax_output;
    try {
        // SoftmaxOp's constructor  takes a vector of
        // inputs, and its forward() requires that vector to have exactly 1
        // element, so wrap just that one
        // tensor in a single-element vector.
        std::vector<std::shared_ptr<Tensor>> softmax_inputs = {model_output};
        auto softmax = std::make_shared<SoftmaxOp>(softmax_inputs);
        softmax_output = softmax->forward();
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Loss::cross_entropy: softmax forward pass failed: ") + e.what());
    }

    std::vector<std::shared_ptr<Tensor>> ce_inputs = {softmax_output, target};

    // Tensor has no default constructor (it always needs a shape), so we
    // can't declare `Tensor loss;` and assign into it inside the try block.
    // An immediately-invoked lambda lets us keep the try/catch context
    // wrapping while direct-initializing `loss` from its return value.
    std::shared_ptr<Tensor> loss = [&]() -> std::shared_ptr<Tensor> {
        try {
            auto cross_entropy_op = std::make_shared<CrossEntropyOp>(ce_inputs);
            return cross_entropy_op->forward();
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Loss::cross_entropy: cross-entropy forward pass failed: ") + e.what());
        }
    }();

    std::shared_ptr<Tensor> mean_loss = [&]() -> std::shared_ptr<Tensor> {
        try {
            std::vector<std::shared_ptr<Tensor>> mean_inputs = {loss};
            auto mean_op = std::make_shared<MeanOp>(mean_inputs);
            return mean_op->forward();
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Loss::cross_entropy: averaging loss failed: ") + e.what());
        };
    }();

    mean_loss->set_grad(std::make_shared<Tensor>(std::vector<size_t>{1}, std::shared_ptr<float[]>(new float[1]{1.0f}), false));
    return mean_loss;
}

std::shared_ptr<Tensor> Loss::mse(std::shared_ptr<Tensor> &model_output, std::shared_ptr<Tensor> &target) {
    validate_inputs(model_output, target);

    auto const &output_data = model_output->data();
    auto const &target_data = target->data();
    size_t size = model_output->size();

    // size == 0 implies target size == 0 too, since validate_inputs()
    // already confirmed the two sizes match.
    if (size == 0) {
        throw std::runtime_error("Loss::mse: input tensors are empty");
    }

    std::shared_ptr<Tensor> loss = [&]() -> std::shared_ptr<Tensor> {
        try {
            auto mse_op = std::make_shared<MseOp>(std::vector<std::shared_ptr<Tensor>>{model_output, target});
            return mse_op->forward();
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Loss::mse: MSE forward pass failed: ") + e.what());
        };
    }();

    std::shared_ptr<Tensor> mean_loss = [&]() -> std::shared_ptr<Tensor> {
        try {
            std::vector<std::shared_ptr<Tensor>> mean_inputs = {loss};
            auto mean_op = std::make_shared<MeanOp>(mean_inputs);
            return mean_op->forward();
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("Loss::mse: averaging loss failed: ") + e.what());
        };
    }();

    mean_loss->set_grad(std::make_shared<Tensor>(std::vector<size_t>{1}, std::shared_ptr<float[]>(new float[1]{1.0f}), false));
    return mean_loss;
}
