

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <iomanip>
#include <iostream>

#include "core/rng.hpp"
#include "core/tensor.hpp"
#include "loss.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"
#include "optimizer/adam.hpp"
#include "optimizer/sgd.hpp"

class MyModel : public Module {

  public:
    explicit MyModel() {
        std::shared_ptr<Linear> linear1 = std::make_shared<Linear>(2, 3);
        std::shared_ptr<Relu> activation1 = std::make_shared<Relu>();

        std::shared_ptr<Linear> linear2 = std::make_shared<Linear>(3, 4);
        std::shared_ptr<Relu> activation2 = std::make_shared<Relu>();

        std::shared_ptr<Linear> linear3 = std::make_shared<Linear>(4, 2);
        layers = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>({linear1, activation1, linear2, activation2, linear3}));
        add_module(layers);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override { return layers->forward(input); }

  private:
    std::shared_ptr<Sequential> layers;
};

std::vector<std::vector<float>> X = {
    // --- TRAIN SET: Batch 1 (Elementi 0-6) ---
    {0.02f, 0.08f}, // Class 0
    {0.95f, 0.03f}, // Class 1
    {0.01f, 0.92f}, // Class 1
    {0.98f, 0.91f}, // Class 0
    {0.05f, 0.04f}, // Class 0
    {0.89f, 0.01f}, // Class 1
    {0.03f, 0.96f}, // Class 1

    // --- TRAIN SET: Batch 2 (Elementi 7-13) ---
    {0.92f, 0.99f}, // Class 0
    {0.07f, 0.01f}, // Class 0
    {0.96f, 0.05f}, // Class 1
    {0.02f, 0.88f}, // Class 1
    {0.88f, 0.94f}, // Class 0
    {0.04f, 0.06f}, // Class 0
    {0.91f, 0.02f}, // Class 1

    // --- TRAIN SET: Batch 3 (Elementi 14-20) ---
    {0.06f, 0.95f}, // Class 1
    {0.97f, 0.93f}, // Class 0
    {0.01f, 0.03f}, // Class 0
    {0.94f, 0.08f}, // Class 1
    {0.08f, 0.97f}, // Class 1
    {0.91f, 0.89f}, // Class 0
    {0.03f, 0.02f}, // Class 0

    // --- TEST SET (Elementi 21-29) ---
    {0.99f, 0.04f}, // Class 1
    {0.05f, 0.92f}, // Class 1
    {0.95f, 0.98f}, // Class 0
    {0.02f, 0.01f}, // Class 0
    {0.91f, 0.06f}, // Class 1
    {0.07f, 0.94f}, // Class 1
    {0.93f, 0.96f}, // Class 0
    {0.04f, 0.05f}, // Class 0
    {0.98f, 0.01f}  // Class 1
};

// Target Y come indici di classe (per Cross-Entropy classica)
std::vector<int> Y_labels = {
    // Train Batch 1
    0, 1, 1, 0, 0, 1, 1,
    // Train Batch 2
    0, 0, 1, 1, 0, 0, 1,
    // Train Batch 3
    1, 0, 0, 1, 1, 0, 0,
    // Test Set
    1, 1, 0, 0, 1, 1, 0, 0, 1};

void training_loop(std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> &target, std::shared_ptr<Tensor> &test_input,
                   std::shared_ptr<Tensor> &test_target, std::shared_ptr<Module> model) {

    size_t num_batches = input->shape()[0];
    size_t batch_size = input->shape()[1];
    size_t features = input->shape()[2];
    size_t elements_per_batch = batch_size * features;
    size_t total_train_samples = num_batches * batch_size;

    // dataloader
    std::vector<std::shared_ptr<Tensor>> batch_inputs;
    std::vector<std::shared_ptr<Tensor>> batch_targets;

    for (size_t b = 0; b < num_batches; ++b) {
        auto b_input = std::make_shared<Tensor>(std::vector<size_t>{batch_size, features});
        auto b_target = std::make_shared<Tensor>(std::vector<size_t>{batch_size, features});
        size_t batch_offset = b * elements_per_batch;
        for (size_t i = 0; i < elements_per_batch; ++i) {
            b_input->set_data(i, input->data()[batch_offset + i]);
            b_target->set_data(i, target->data()[batch_offset + i]);
        }
        batch_inputs.push_back(b_input);
        batch_targets.push_back(b_target);
    }

    size_t epochs = 120;
    SGD optimizer(0.2f, model->parameters()); // (42) (12)
    // Adam optimizer(0.001f, model->parameters());

    for (size_t i = 0; i < epochs; ++i) {
        float train_accuracy = 0.0f;
        float train_loss = 0.0f;
        float test_accuracy = 0.0f;
        float test_loss = 0.0f;

        model->set_train();

        for (size_t j = 0; j < num_batches; ++j) {
            optimizer.zero_grad();

            std::shared_ptr<Tensor> prediction = model->forward(batch_inputs[j]);
            std::shared_ptr<Tensor> loss = Loss::cross_entropy(prediction, batch_targets[j]);
            train_loss += loss->item();

            const float *raw_ptr = prediction->data().get();
            for (size_t k = 0; k < batch_size; ++k) {
                size_t offset_feature = k * 2;
                const float *max_it = std::max_element(raw_ptr + offset_feature, raw_ptr + offset_feature + 2);
                size_t max_idx = std::distance(raw_ptr, max_it);
                if (batch_targets[j]->data()[max_idx] == 1) {
                    train_accuracy++;
                }
            }

            loss->backward();
            optimizer.step();
        }

        train_loss = train_loss / num_batches;
        train_accuracy = train_accuracy / static_cast<float>(total_train_samples);

        model->set_eval();

        std::shared_ptr<Tensor> prediction = model->forward(test_input);
        std::shared_ptr<Tensor> loss = Loss::cross_entropy(prediction, test_target);
        test_loss = loss->item();

        const float *raw_ptr = prediction->data().get();
        size_t evaluation_samples = test_input->size() / features;
        for (size_t k = 0; k < evaluation_samples; ++k) {
            size_t offset_feature = k * 2;
            const float *max_it = std::max_element(raw_ptr + offset_feature, raw_ptr + offset_feature + 2);
            size_t max_idx = std::distance(raw_ptr, max_it);
            if (test_target->data()[max_idx] == 1) {
                test_accuracy++;
            }
        }
        test_accuracy = test_accuracy / static_cast<float>(evaluation_samples);

        if ((i + 1) % 10 == 0) {
            std::cout << std::fixed << std::setprecision(4) << "[Epoch " << (i + 1) << "/" << epochs << "] "
                      << "train_loss: " << train_loss << " | train_acc: " << train_accuracy << " | test_loss: " << test_loss
                      << " | test_acc: " << test_accuracy << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc > 2) {
        std::cerr << "Usage: XORTest [seed]" << std::endl;
        return 1;
    }

    if (argc == 2) {
        try {
            size_t parsed_characters = 0;
            unsigned long parsed_seed = std::stoul(argv[1], &parsed_characters);
            if (parsed_characters != std::string(argv[1]).size() || parsed_seed > std::numeric_limits<unsigned int>::max()) {
                throw std::invalid_argument("seed fuori intervallo");
            }
            rng::set_seed(static_cast<unsigned int>(parsed_seed));
        } catch (const std::exception &) {
            std::cerr << "Seed non valido. Usa un intero tra 0 e " << std::numeric_limits<unsigned int>::max() << "." << std::endl;
            return 1;
        }
    }

    // dataset
    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(std::vector<size_t>{3, 7, 2});
    for (size_t i = 0; i < input->size(); ++i) {
        size_t j = i / 2;
        size_t k = i % 2;
        input->set_data(i, X[j][k]);
    }

    std::shared_ptr<Tensor> target = std::make_shared<Tensor>(std::vector<size_t>{3, 7, 2});
    for (size_t i = 0; i < 21; ++i) {
        size_t offset = i * 2;
        if (Y_labels[i] == 0) {
            target->set_data(offset, 1);
            target->set_data(offset + 1, 0);
        } else {
            target->set_data(offset, 0);
            target->set_data(offset + 1, 1);
        }
    }

    size_t test_size = X.size() - 21; // 9
    std::shared_ptr<Tensor> test_input = std::make_shared<Tensor>(std::vector<size_t>{test_size, 2});
    std::shared_ptr<Tensor> test_target = std::make_shared<Tensor>(std::vector<size_t>{test_size, 2});
    for (size_t i = 0; i < test_size; ++i) {
        size_t src = 21 + i;
        test_input->set_data(i * 2, X[src][0]);
        test_input->set_data(i * 2 + 1, X[src][1]);

        size_t offset = i * 2;
        if (Y_labels[src] == 0) {
            test_target->set_data(offset, 1);
            test_target->set_data(offset + 1, 0);
        } else {
            test_target->set_data(offset, 0);
            test_target->set_data(offset + 1, 1);
        }
    }

    auto model = std::make_shared<MyModel>();
    model->init_he();
    training_loop(input, target, test_input, test_target, model);

    return 0;
}
