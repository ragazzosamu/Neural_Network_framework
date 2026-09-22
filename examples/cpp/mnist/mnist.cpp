#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "loss.hpp"
#include "mnist_dataloader.hpp"
#include "nn/conv2D.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"
#include "ops/flatten_convolution_output.hpp"
#include "optimizer/adam.hpp"

namespace fs = std::filesystem;

class Convolutional_Model : public Module {
  public:
    explicit Convolutional_Model() {
        auto conv1 = std::make_shared<Conv2D>(1, 16, 5, 5, 2, 1);
        auto activation1 = std::make_shared<Relu>();
        auto conv2 = std::make_shared<Conv2D>(16, 32, 3, 3, 1, 2);
        auto activation2 = std::make_shared<Relu>();
        auto conv3 = std::make_shared<Conv2D>(32, 64, 3, 3, 1, 2);
        auto activation3 = std::make_shared<Relu>();
        conv_layers =
            std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>({conv1, activation1, conv2, activation2, conv3, activation3}));
        add_module(conv_layers);
        output_layer = std::make_shared<Linear>(64 * 7 * 7, 10);
        add_module(output_layer);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override {
        if (!input || input->shape().size() != 4) {
            throw std::invalid_argument("Convolutional_Model expects input batches shaped [N, C, H, W]");
        }
        const auto conv_output = conv_layers->forward(input);
        const auto &shape = conv_output->shape();
        if (shape.size() != 4 || shape[1] * shape[2] * shape[3] != 64 * 7 * 7) {
            throw std::runtime_error("Convolutional_Model produced an unexpected convolution output shape");
        }
        auto flattened_output = std::make_shared<FlattenConvolutionOutputOp>(conv_output)->forward();
        return output_layer->forward(flattened_output);
    }

  private:
    std::shared_ptr<Sequential> conv_layers;
    std::shared_ptr<Linear> output_layer;
};

class Linear_Model : public Module {
  public:
    explicit Linear_Model() {
        auto linear1 = std::make_shared<Linear>(784, 64);
        auto activation1 = std::make_shared<Relu>();
        auto linear2 = std::make_shared<Linear>(64, 10);
        layers = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>({linear1, activation1, linear2}));
        add_module(layers);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override { return layers->forward(input); }

  private:
    std::shared_ptr<Sequential> layers;
};

void training_loop(const std::vector<std::shared_ptr<Tensor>> &input, std::vector<std::shared_ptr<Tensor>> &target,
                   const std::vector<std::shared_ptr<Tensor>> &test_input, std::vector<std::shared_ptr<Tensor>> &test_target,
                   const std::string &model_name, std::shared_ptr<Module> model) {
    constexpr std::size_t epochs = 10;
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
        for (std::size_t b = 0; b < input.size(); ++b) {
            optimizer.zero_grad();
            auto prediction = model->forward(input[b]);
            auto loss = Loss::cross_entropy(prediction, target[b]);
            train_loss += loss->item();
            const float *raw_ptr = prediction->data().get();
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
        train_loss /= input.size();
        std::size_t train_samples = 0;
        for (const auto &batch : input)
            train_samples += batch->shape()[0];
        train_accuracy /= static_cast<float>(train_samples);

        model->set_eval();
        for (std::size_t b = 0; b < test_input.size(); ++b) {
            auto prediction = model->forward(test_input[b]);
            auto loss = Loss::cross_entropy(prediction, test_target[b]);
            test_loss += loss->item();
            const float *raw_ptr = prediction->data().get();
            for (std::size_t k = 0; k < test_input[b]->shape()[0]; ++k) {
                const std::size_t offset = k * 10;
                const float *max_it = std::max_element(raw_ptr + offset, raw_ptr + offset + 10);
                if (test_target[b]->data()[k * 10 + static_cast<std::size_t>(std::distance(raw_ptr + offset, max_it))] == 1.0f) {
                    test_accuracy++;
                }
            }
        }
        test_loss /= test_input.size();
        std::size_t test_samples = 0;
        for (const auto &batch : test_input)
            test_samples += batch->shape()[0];
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

int main() {
    try {
        constexpr std::size_t batch_size = 2048;
        const fs::path project_root = fs::path(NEURAL_NETWORK_PROJECT_ROOT);
        MnistDatasetLoader loader(project_root / "examples" / "data" / "mnist");
        loader.load_training_set();
        loader.load_test_set();
        const auto &train_tensor = loader.training_images();
        const auto &test_tensor = loader.test_images();
        const auto &train_labels = loader.training_labels();
        const auto &test_labels = loader.test_labels();
        auto train_batches = loader.make_batches(batch_size);
        auto test_batches = loader.make_batches(batch_size, false);
        auto train_flattened = loader.flatten();
        auto test_flattened = loader.flatten(false);
        auto train_flattened_batches = loader.make_batches(batch_size);
        auto test_flattened_batches = loader.make_batches(batch_size, false);
        loader.unflatten(28, 28);
        loader.unflatten(28, 28, false);

        std::cout << "train_tensor shape: [" << train_tensor->shape()[0] << ", " << train_tensor->shape()[1] << ", " << train_tensor->shape()[2]
                  << ", " << train_tensor->shape()[3] << "]\n";
        std::cout << "test_tensor shape: [" << test_tensor->shape()[0] << ", " << test_tensor->shape()[1] << ", " << test_tensor->shape()[2] << ", "
                  << test_tensor->shape()[3] << "]\n";
        std::cout << "train_flattened shape: [" << train_flattened->shape()[0] << ", " << train_flattened->shape()[1] << ", "
                  << train_flattened->shape()[2] << "]\n";
        std::cout << "test_flattened shape: [" << test_flattened->shape()[0] << ", " << test_flattened->shape()[1] << ", "
                  << test_flattened->shape()[2] << "]\n";
        std::cout << "train_labels shape: [" << train_labels->shape()[0] << ", " << train_labels->shape()[1] << "]\n";
        std::cout << "test_labels shape: [" << test_labels->shape()[0] << ", " << test_labels->shape()[1] << "]\n";
        std::cout << "First sample normalized pixel: " << std::fixed << std::setprecision(4) << train_tensor->data()[0] << "\n";
        std::cout << "Train image batches: " << train_batches.images.size() << "\n";
        std::cout << "Train label batches: " << train_batches.labels.size() << "\n";
        std::cout << "Test image batches: " << test_batches.images.size() << "\n";
        std::cout << "Test label batches: " << test_batches.labels.size() << "\n";

        std::cout << "Starting Convolutional_Model training...\n";
        auto convolutional_model = std::make_shared<Convolutional_Model>();
        convolutional_model->init_he();
        training_loop(train_batches.images, train_batches.labels, test_batches.images, test_batches.labels, "Convolutional_Model",
                      convolutional_model);

        std::cout << "Starting Linear_Model training...\n";
        auto linear_model = std::make_shared<Linear_Model>();
        linear_model->init_he();
        training_loop(train_flattened_batches.images, train_flattened_batches.labels, test_flattened_batches.images, test_flattened_batches.labels,
                      "Linear_Model", linear_model);

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "MNIST loader error: " << ex.what() << std::endl;
        return 1;
    }
}