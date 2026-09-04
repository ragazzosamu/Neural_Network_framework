#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <exception> // std::rethrow_if_nested
#include <memory>
#include <vector>

#include "core/tensor.hpp"
#include "nn/linear.hpp"
#include "nn/module.hpp"
#include "nn/relu.hpp"
#include "nn/sequential.hpp"

namespace {

// Builds the same small network in every test case so we don't repeat the
// boilerplate. Implemented as a real Module subclass (mirrors how Sequential
// itself is built) rather than a plain struct wrapping one.
class TestNetwork : public Module {
  public:
    TestNetwork()
        : linear1(std::make_shared<Linear>(4, 8)), relu(std::make_shared<Relu>()), linear2(std::make_shared<Linear>(8, 2)),
          sequential(std::make_shared<Sequential>(std::vector<std::shared_ptr<Module>>{linear1, relu, linear2})) {
        add_module(sequential);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override { return sequential->forward(input); }

    std::shared_ptr<Linear> linear1;
    std::shared_ptr<Relu> relu;
    std::shared_ptr<Linear> linear2;
    std::shared_ptr<Sequential> sequential;
};

// Network with deliberately mismatched sizes between linear1's output (8)
// and linear2's input (5). Also a real Module subclass.
class MismatchedNetwork : public Module {
  public:
    MismatchedNetwork()
        : linear1_(std::make_shared<Linear>(4, 8)), relu_(std::make_shared<Relu>()),
          linear2_(std::make_shared<Linear>(5, 2)) { // <-- mismatched: 8 != 5
        add_module(linear1_);
        add_module(relu_);
        add_module(linear2_);
    }

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override {
        auto out = linear1_->forward(input);
        out = relu_->forward(out);
        out = linear2_->forward(out);
        return out;
    }

  private:
    std::shared_ptr<Linear> linear1_;
    std::shared_ptr<Relu> relu_;
    std::shared_ptr<Linear> linear2_;
};

} // namespace

// ---------------------------------------------------------------------------
// Main check: forward pass through Sequential, plus correct aggregation of
// parameters() and modules(). No checks on output values here — the goal is
// only to confirm that the layer chaining in forward() doesn't blow up and
// that Sequential doesn't lose references while collecting parameters and
// sub-modules from its children.
// ---------------------------------------------------------------------------
TEST_CASE("Sequential: forward pass chains layers without throwing", "[nn][sequential]") {
    TestNetwork net;

    // Linear expects input shaped {batch, 1, input_size}; linear1 takes 4
    // features, so the last dimension here must be 4.
    auto input = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 4}); // batch=3, features=4

    std::shared_ptr<Tensor> output;
    REQUIRE_NOTHROW(output = net.forward(input));
    REQUIRE(output != nullptr);

    // Output of the whole network should be {batch, 1, output_size} of the
    // last layer, i.e. {3, 1, 2}.
    REQUIRE(output->shape() == std::vector<size_t>{3, 1, 2});
}

TEST_CASE("Sequential: parameters() collects every sub-module's parameters", "[nn][sequential]") {
    TestNetwork net;

    // Relu has no parameters of its own: expected parameters come only from
    // the two Linear layers.
    auto expected = net.linear1->parameters();
    auto linear2_params = net.linear2->parameters();
    expected.insert(expected.end(), linear2_params.begin(), linear2_params.end());

    auto actual = net.sequential->parameters();

    REQUIRE(actual.size() == expected.size());

    // Size alone isn't enough: check these are the SAME shared_ptr (same
    // reference), not copies or the wrong tensors.
    for (const auto &param : expected) {
        auto found = std::find(actual.begin(), actual.end(), param);
        REQUIRE(found != actual.end());
    }
}

TEST_CASE("Sequential: modules() returns every sub-module, in order", "[nn][sequential]") {
    TestNetwork net;

    auto actual = net.sequential->modules();

    REQUIRE(actual.size() == 3);

    // Same idea as above: pointer identity, not just "some Linear".
    REQUIRE(std::find(actual.begin(), actual.end(), std::static_pointer_cast<Module>(net.linear1)) != actual.end());
    REQUIRE(std::find(actual.begin(), actual.end(), std::static_pointer_cast<Module>(net.relu)) != actual.end());
    REQUIRE(std::find(actual.begin(), actual.end(), std::static_pointer_cast<Module>(net.linear2)) != actual.end());

    // Insertion order should match the order passed to the constructor.
    REQUIRE(actual[0] == std::static_pointer_cast<Module>(net.linear1));
    REQUIRE(actual[1] == std::static_pointer_cast<Module>(net.relu));
    REQUIRE(actual[2] == std::static_pointer_cast<Module>(net.linear2));
}

TEST_CASE("Module::add_module rejects a null sub-module", "[nn][module]") {
    Sequential sequential({});
    REQUIRE_THROWS_AS(sequential.add_module(nullptr), std::invalid_argument);
}

TEST_CASE("Linear::forward: shapes are correct and errors are wrapped as std::runtime_error", "[nn][linear]") {

    SECTION("produces the expected output shape") {
        Linear linear(4, 8);
        auto input = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 4}); // batch=3, features=4

        std::shared_ptr<Tensor> output;
        REQUIRE_NOTHROW(output = linear.forward(input));
        REQUIRE(output->shape() == std::vector<size_t>{3, 1, 8});
    }

    SECTION("throws on a mismatched input dimension") {
        Linear linear(4, 8);
        auto bad_input = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 5}); // expected 4, got 5

        REQUIRE_THROWS_AS(linear.forward(bad_input), std::runtime_error);
    }

    SECTION("throws on a null input") {
        Linear linear(4, 8);

        REQUIRE_THROWS_AS(linear.forward(nullptr), std::runtime_error);
    }

    SECTION("preserves the original MatMulOp/MatSumOp exception as a nested exception") {
        Linear linear(4, 8);
        auto bad_input = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 5}); // expected 4, got 5

        try {
            linear.forward(bad_input);
            FAIL("Expected Linear::forward to throw for a mismatched input shape");
        } catch (const std::exception &outer) {
            // outer is the "Linear error: ..." wrapper. Unwrap it to check the
            // real error coming from MatMulOp is still there.
            try {
                std::rethrow_if_nested(outer);
                FAIL("Expected a nested exception from MatMulOp/MatSumOp, found none");
            } catch (const std::invalid_argument &inner) {
                INFO("Original error preserved: " << inner.what());
                SUCCEED();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Networks whose layer sizes don't line up, either between two consecutive
// layers or between the input tensor and the first layer. Since we now know
// Linear::forward wraps every internal failure as std::runtime_error (with
// the original error preserved as a nested exception), we can assert on that
// directly instead of just logging a WARN.
// ---------------------------------------------------------------------------
TEST_CASE("Sequential network with mismatched layer sizes", "[nn][edge-cases]") {

    SECTION("Mismatch between linear1 output and linear2 input") {
        MismatchedNetwork net;                                               // linear1: 4 -> 8, linear2 expects 5 instead of 8
        auto input = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 4}); // batch=3, features=4

        REQUIRE_THROWS_AS(net.forward(input), std::runtime_error);
    }

    SECTION("Mismatch between linear1 output and linear2 input (message content)") {
        MismatchedNetwork net;
        auto input = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 4});

        try {
            net.forward(input);
            FAIL("Expected MismatchedNetwork::forward to throw for mismatched linear1/linear2 sizes");
        } catch (const std::runtime_error &e) {
            REQUIRE_THAT(std::string(e.what()), Catch::Matchers::ContainsSubstring("Linear error:"));
        }
    }

    SECTION("Mismatch between input tensor shape and first layer's input size") {
        TestNetwork net;                                                     // linear1: 4 -> 8, relu, linear2: 8 -> 2
        auto input = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 6}); // expected 4, got 6

        REQUIRE_THROWS_AS(net.forward(input), std::runtime_error);
    }
}