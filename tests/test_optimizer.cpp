#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

#include "core/tensor.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/sequential.hpp"
#include "optimizer/adam.hpp"
#include "optimizer/sgd.hpp"

using Catch::Approx;

namespace {

std::shared_ptr<Tensor> make_tensor(const std::vector<size_t> &shape, const std::vector<float> &values) {
    auto tensor = std::make_shared<Tensor>(shape);
    REQUIRE(tensor->size() == values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        tensor->set_data(i, values[i]);
    }
    return tensor;
}

void backward_graph(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &grad) {
    const auto &operation = output->get_operation();
    REQUIRE(operation != nullptr);
    operation->backward(grad);

    for (const auto &input : operation->inputs()) {
        if (input->get_operation() != nullptr) {
            backward_graph(input, input->get_grad());
        }
    }
}

class TwoLayerGraph : public Module {
  public:
    std::shared_ptr<Linear> linear1 = std::make_shared<Linear>(2, 2);
    std::shared_ptr<Linear> linear2 = std::make_shared<Linear>(2, 1);
    std::shared_ptr<Sequential> sequential;
    mutable std::shared_ptr<Tensor> output;

    TwoLayerGraph() {
        sequential = std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{linear1, linear2});
        add_module(sequential);

        auto weights1 = linear1->parameters()[0];
        auto weights2 = linear2->parameters()[0];

        const std::vector<float> weights1_values{1.0f, 2.0f, 3.0f, 4.0f};
        for (size_t i = 0; i < weights1->size(); ++i) {
            weights1->set_data(i, weights1_values[i]);
        }
        const std::vector<float> weights2_values{5.0f, 6.0f};
        for (size_t i = 0; i < weights2->size(); ++i) {
            weights2->set_data(i, weights2_values[i]);
        }
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input_tensor) const override {
        output = sequential->forward(input_tensor);
        return output;
    }

    void backward() { backward_graph(output, make_tensor({1, 1}, {1.0f})); }
};

void require_values(const std::shared_ptr<Tensor> &tensor, const std::vector<float> &expected) {
    REQUIRE(tensor->size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(tensor->data()[i] == Approx(expected[i]));
    }
}

} // namespace

TEST_CASE("SGD and Adam reject parameters without gradients", "[optimizer][exceptions]") {
    auto parameter = make_tensor({2}, {1.0f, 2.0f});

    SECTION("SGD") {
        SGD optimizer(0.1f, {parameter});
        REQUIRE_THROWS_AS(optimizer.step(), std::runtime_error);
    }

    SECTION("Adam") {
        Adam optimizer(0.1f, {parameter});
        REQUIRE_THROWS_AS(optimizer.step(), std::runtime_error);
    }
}

TEST_CASE("Two-layer graph propagates gradients and SGD updates parameters", "[optimizer][sgd][integration]") {
    TwoLayerGraph graph;
    auto input = make_tensor({1, 2}, {1.0f, 2.0f});
    graph.forward(input);

    REQUIRE(graph.output != nullptr);
    require_values(graph.output, {95.0f});

    graph.backward();

    auto parameters = graph.parameters();
    REQUIRE(parameters.size() == 4);
    require_values(parameters[0]->get_grad(), {5.0f, 6.0f, 10.0f, 12.0f});
    require_values(parameters[1]->get_grad(), {5.0f, 6.0f});
    require_values(parameters[2]->get_grad(), {7.0f, 10.0f});
    require_values(parameters[3]->get_grad(), {1.0f});

    SGD optimizer(0.1f, graph.parameters());
    REQUIRE_NOTHROW(optimizer.step());

    require_values(parameters[0], {0.5f, 1.4f, 2.0f, 2.8f});
    require_values(parameters[1], {-0.5f, -0.6f});
    require_values(parameters[2], {4.3f, 5.0f});
    require_values(parameters[3], {-0.1f});
}

TEST_CASE("Adam performs its first normalized update", "[optimizer][adam]") {
    auto parameter = make_tensor({2}, {1.0f, 2.0f});
    parameter->set_grad(make_tensor({2}, {2.0f, -4.0f}));

    Adam optimizer(0.1f, {parameter});
    REQUIRE_NOTHROW(optimizer.step());

    require_values(parameter, {0.68377f, 2.31623f});
}

TEST_CASE("zero_grad clears every optimized parameter gradient", "[optimizer]") {
    TwoLayerGraph graph;
    graph.forward(make_tensor({1, 2}, {1.0f, 2.0f}));
    graph.backward();

    SGD optimizer(0.1f, graph.parameters());
    for (const auto &parameter : graph.parameters()) {
        REQUIRE(parameter->get_grad() != nullptr);
    }

    REQUIRE_NOTHROW(optimizer.zero_grad());
    for (const auto &parameter : graph.parameters()) {
        REQUIRE(parameter->get_grad() == nullptr);
    }
}
