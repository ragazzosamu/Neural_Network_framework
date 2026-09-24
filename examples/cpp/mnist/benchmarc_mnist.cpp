#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/rng.hpp"
#include "loss.hpp"
#include "mnist_dataloader.hpp"
#include "nn/conv2D.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"
#include "ops/flatten_convolution_output.hpp"
#include "optimizer/adam.hpp"
#include <cstdlib>
namespace {
std::size_t env_size(const char *name, std::size_t fallback) {
    const char *v = std::getenv(name);
    return v ? static_cast<std::size_t>(std::stoul(v)) : fallback;
}

class BenchmarkLinearModel final : public Module {
  public:
    BenchmarkLinearModel() {
        layers_ = std::make_shared<Sequential>(
            std::vector<std::shared_ptr<Module>>{std::make_shared<Linear>(784, 64), std::make_shared<Relu>(), std::make_shared<Linear>(64, 10)});
        add_module(layers_);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override { return layers_->forward(input); }

  private:
    std::shared_ptr<Sequential> layers_;
};

class BenchmarkConvolutionalModel final : public Module {
  public:
    BenchmarkConvolutionalModel() {
        conv_layers_ = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{
            std::make_shared<Conv2D>(1, 16, 5, 5, 2, 1), std::make_shared<Relu>(), std::make_shared<Conv2D>(16, 32, 3, 3, 1, 2),
            std::make_shared<Relu>(), std::make_shared<Conv2D>(32, 64, 3, 3, 1, 2), std::make_shared<Relu>()});
        output_layer_ = std::make_shared<Linear>(64 * 7 * 7, 10);
        add_module(conv_layers_);
        add_module(output_layer_);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override {
        const auto output = conv_layers_->forward(input);
        return output_layer_->forward(std::make_shared<FlattenConvolutionOutputOp>(output)->forward());
    }

  private:
    std::shared_ptr<Sequential> conv_layers_;
    std::shared_ptr<Linear> output_layer_;
};

void training_loop(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &target,
                   const std::vector<std::shared_ptr<Tensor>> &test_input, std::vector<std::shared_ptr<Tensor>> &test_target,
                   const std::string &model_name, std::shared_ptr<Module> model) {

    const std::size_t max_batches = env_size("BENCH_MAX_BATCHES", static_cast<std::size_t>(-1));
    const std::size_t n_train = std::min(input.size(), max_batches);
    const std::size_t n_test = std::min(test_input.size(), max_batches);

    constexpr std::size_t epochs = 1;
    Adam optimizer(0.001f, model->parameters());
    float final_train_loss = 0.0f;
    float final_test_loss = 0.0f;
    float final_train_accuracy = 0.0f;
    float final_test_accuracy = 0.0f;
    for (std::size_t e = 0; e < epochs; ++e) {
        float train_loss = 0.0f;
        float test_loss = 0.0f;
        float train_accuracy = 0.0f;
        float test_accuracy = 0.0f;
        model->set_train();
        for (std::size_t b = 0; b < n_train; ++b) {
            optimizer.zero_grad();
            auto prediction = model->forward(input[b]);
            auto loss = Loss::cross_entropy(prediction, target[b]);
            train_loss += loss->item();
            const float *raw_ptr = prediction->data();
            for (std::size_t k = 0; k < input[b]->shape()[0]; ++k) {
                const std::size_t offset = k * 10;
                const float *max_it = std::max_element(raw_ptr + offset, raw_ptr + offset + 10);
                if (target[b]->data()[k * 10 + static_cast<std::size_t>(std::distance(raw_ptr + offset, max_it))] == 1.0f) {
                    train_accuracy++;
                }
            }
            loss->backward();
            optimizer.step();
        }
        train_loss /= static_cast<float>(n_train);
        std::size_t train_samples = 0;
        for (std::size_t b = 0; b < n_train; ++b)
            train_samples += input[b]->shape()[0];
        train_accuracy /= static_cast<float>(train_samples);

        model->set_eval();
        for (std::size_t b = 0; b < n_test; ++b) {
            auto prediction = model->forward(test_input[b]);
            auto loss = Loss::cross_entropy(prediction, test_target[b]);
            test_loss += loss->item();
            const float *raw_ptr = prediction->data();
            for (std::size_t k = 0; k < test_input[b]->shape()[0]; ++k) {
                const std::size_t offset = k * 10;
                const float *max_it = std::max_element(raw_ptr + offset, raw_ptr + offset + 10);
                if (test_target[b]->data()[k * 10 + static_cast<std::size_t>(std::distance(raw_ptr + offset, max_it))] == 1.0f) {
                    test_accuracy++;
                }
            }
        }
        test_loss /= static_cast<float>(n_test);
        std::size_t test_samples = 0;
        for (std::size_t b = 0; b < n_test; ++b)
            test_samples += test_input[b]->shape()[0];
        test_accuracy /= static_cast<float>(test_samples);

        final_train_loss = train_loss;
        final_test_loss = test_loss;
        final_train_accuracy = train_accuracy;
        final_test_accuracy = test_accuracy;
        std::cout << std::fixed << std::setprecision(4) << "[" << model_name << "] Epoch " << (e + 1) << "/" << epochs
                  << " | train_loss: " << train_loss << " | train_acc: " << train_accuracy << " | test_loss: " << test_loss
                  << " | test_acc: " << test_accuracy << "\n";
    }
    std::cout << std::fixed << std::setprecision(4) << "Final " << model_name << " results | train_loss: " << final_train_loss
              << " | train_acc: " << final_train_accuracy << " | test_loss: " << final_test_loss << " | test_acc: " << final_test_accuracy << "\n";
}

// The weight initialization is the only random step of the benchmark (the
// batches are not shuffled). With BENCH_SEED set, every run and every build
// (e.g. CPU and GPU) starts from the same weights, so their losses and
// accuracies can be compared directly. The seed is set again before each model,
// so a model's weights don't depend on which models were trained before it.
// Without BENCH_SEED, a random seed is used as before.
void seed_from_env() {
    if (const char *seed = std::getenv("BENCH_SEED")) {
        rng::set_seed(static_cast<unsigned int>(std::stoul(seed)));
    }
}

template <typename Model> void train_benchmark(MnistDatasetLoader::BatchTensors &train_batches, MnistDatasetLoader::BatchTensors &test_batches) {
    seed_from_env();
    auto model = std::make_shared<Model>();
    model->init_he();
    training_loop(train_batches.images, train_batches.labels, test_batches.images, test_batches.labels, "BENCHMARK", model);
}

std::pair<MnistDatasetLoader::BatchTensors, MnistDatasetLoader::BatchTensors> make_batches(bool convolutional) {
    MnistDatasetLoader loader;
    loader.load_training_set();
    loader.load_test_set();
    constexpr std::size_t batch_size = 2048;
    if (convolutional) {
        auto train = loader.make_batches(batch_size);
        auto test = loader.make_batches(batch_size, false);
        return {std::move(train), std::move(test)};
    }
    loader.flatten();
    auto train = loader.make_batches(batch_size);
    loader.flatten(false);
    auto test = loader.make_batches(batch_size, false);
    return {std::move(train), std::move(test)};
}

} // namespace

TEST_CASE("MNIST training benchmark") {
    const char *which_env = std::getenv("BENCH_MODEL");
    const std::string which = which_env ? which_env : "all";

    if (which == "all" || which == "linear") {
        auto linear_batches = make_batches(false);
        const auto start = std::chrono::steady_clock::now();
        train_benchmark<BenchmarkLinearModel>(linear_batches.first, linear_batches.second);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "Tempo totale modello lineare: " << elapsed << "s\n";
    }

    if (which == "all" || which == "conv") {
        auto convolutional_batches = make_batches(true);
        const auto start = std::chrono::steady_clock::now();
        train_benchmark<BenchmarkConvolutionalModel>(convolutional_batches.first, convolutional_batches.second);
        const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "Tempo totale modello convoluzionale: " << elapsed << "s\n";
    }
}
