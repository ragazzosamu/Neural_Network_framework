#include "loss.hpp";

#include <stdexcept>
#include <string>

Loss::Loss(std::vector<std::shared_ptr<Tensor>> inputs) : l_inputs(std::move(inputs)) {
    if (l_inputs.size() != 2) {
        throw std::invalid_argument("Loss: expected exactly 2 inputs (model output, target), got " + std::to_string(l_inputs.size()));
    }
    if (!l_inputs[0] || !l_inputs[1]) {
        throw std::invalid_argument("Loss: model output and target tensors must not be null");
    }
}

void Loss::validate_inputs() const {
    // These first two checks are defensive: the constructor already
    // guarantees them, but re-checking here keeps each public method
    // self-contained and safe even if that invariant is ever relaxed.
    if (!l_inputs[0] || !l_inputs[1]) {
        throw std::invalid_argument("Loss: model output and target tensors must not be null");
    }
    // Shape equality (not just matching total size) mirrors the check
    // CrossEntropyOp::forward() does internally, and also catches cases
    // like {2,3} vs {3,2} that have the same element count but are not
    // actually compatible.
    if (l_inputs[0]->shape() != l_inputs[1]->shape()) {
        throw std::invalid_argument("Loss: model output and target shape mismatch (sizes " + std::to_string(l_inputs[0]->size()) + " vs " +
                                    std::to_string(l_inputs[1]->size()) + ")");
    }
}

std::shared_ptr<Tensor> Loss::cross_entropy() {
    validate_inputs();

    auto const &model_output = l_inputs[0];
    auto const &target = l_inputs[1];

    std::shared_ptr<Tensor> softmax_output;
    try {
        // SoftmaxOp's constructor  takes a vector of
        // inputs, and its forward() requires that vector to have exactly 1
        // element. Passing l_inputs directly would compile (same type) but
        // throw at runtime, since l_inputs has 2 elements (output, target).
        // Softmax only ever needs the model output, so wrap just that one
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
    return mean_loss;
}

std::shared_ptr<Tensor> Loss::mse() {
    validate_inputs();

    auto const &model_output = l_inputs[0];
    auto const &target = l_inputs[1];

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
    return mean_loss;
}