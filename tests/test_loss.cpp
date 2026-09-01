// Catch2 v3 test suite for the Loss class.
//
// Based on the usage seen in softmax.cpp, Tensor's constructor only takes
// a shape (`Tensor output(shape);`), and individual elements are set with
// `set_data(index, value)`. `make_tensor()` below builds tensors that way.
// If your real API differs, that's the only place you should need to change.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "loss.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

std::shared_ptr<Tensor> make_tensor(std::vector<size_t> shape, const std::vector<float> &data) {
    auto tensor = std::make_shared<Tensor>(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        tensor->set_data(i, data[i]);
    }
    return tensor;
}

} // namespace

// ---------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------

TEST_CASE("Loss constructor rejects a wrong number of inputs", "[loss][exceptions]") {
    auto t = make_tensor({2}, {1.0f, 2.0f});

    SECTION("empty input list") { REQUIRE_THROWS_AS(Loss(std::vector<std::shared_ptr<Tensor>>{}), std::invalid_argument); }

    SECTION("single input") { REQUIRE_THROWS_AS(Loss(std::vector<std::shared_ptr<Tensor>>{t}), std::invalid_argument); }

    SECTION("three inputs") { REQUIRE_THROWS_AS(Loss(std::vector<std::shared_ptr<Tensor>>{t, t, t}), std::invalid_argument); }
}

TEST_CASE("Loss constructor rejects null tensors", "[loss][exceptions]") {
    auto t = make_tensor({2}, {1.0f, 2.0f});
    std::shared_ptr<Tensor> null_tensor;

    REQUIRE_THROWS_AS(Loss(std::vector<std::shared_ptr<Tensor>>{null_tensor, t}), std::invalid_argument);
    REQUIRE_THROWS_AS(Loss(std::vector<std::shared_ptr<Tensor>>{t, null_tensor}), std::invalid_argument);
}

// ---------------------------------------------------------------------
// mse()
// ---------------------------------------------------------------------

TEST_CASE("mse computes the correct value for known inputs", "[loss][mse]") {
    auto output = make_tensor({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto target = make_tensor({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    // diffs: 0, 1, 2, 3 -> squared: 0, 1, 4, 9 -> mean = 14 / 4 = 3.5
    Loss loss({output, target});

    REQUIRE(loss.mse() == Catch::Approx(3.5f));
}

TEST_CASE("mse is zero when output equals target", "[loss][mse]") {
    auto output = make_tensor({3}, {0.5f, -1.0f, 2.0f});
    auto target = make_tensor({3}, {0.5f, -1.0f, 2.0f});
    Loss loss({output, target});

    REQUIRE(loss.mse() == Catch::Approx(0.0f));
}

TEST_CASE("mse throws on size mismatch between output and target", "[loss][mse][exceptions]") {
    auto output = make_tensor({3}, {1.0f, 2.0f, 3.0f});
    auto target = make_tensor({2}, {1.0f, 2.0f});
    Loss loss({output, target});

    REQUIRE_THROWS_AS(loss.mse(), std::invalid_argument);
}

TEST_CASE("mse throws on empty tensors instead of dividing by zero", "[loss][mse][exceptions]") {
    auto output = make_tensor({0}, {});
    auto target = make_tensor({0}, {});
    Loss loss({output, target});

    REQUIRE_THROWS_AS(loss.mse(), std::runtime_error);
}

// ---------------------------------------------------------------------
// cross_entropy()
// ---------------------------------------------------------------------

TEST_CASE("cross_entropy throws on size mismatch between output and target", "[loss][cross_entropy][exceptions]") {
    auto output = make_tensor({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
    auto target = make_tensor({2}, {0.0f, 1.0f}); // wrong element count on purpose
    Loss loss({output, target});

    REQUIRE_THROWS_AS(loss.cross_entropy(), std::invalid_argument);
}

TEST_CASE("cross_entropy throws on same total size but different shape", "[loss][cross_entropy][exceptions]") {
    // Both have 6 elements, but {2,3} vs {3,2} are not compatible: this
    // must be caught by Loss's own shape check, not just a size check.
    auto output = make_tensor({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});
    auto target = make_tensor({3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
    Loss loss({output, target});

    REQUIRE_THROWS_AS(loss.cross_entropy(), std::invalid_argument);
}

TEST_CASE("cross_entropy matches the hand-computed value for a one-hot target", "[loss][cross_entropy]") {
    // Batch of 1, 3 classes, one-hot target on class 0.
    // Expected value computed independently in Python (double precision)
    // from the same formula as CrossEntropyOp::forward():
    //   softmax(output) -> p, then loss = -log(p[0] + 1e-7)
    // p = [0.6590011389, 0.2424329707, 0.0985658904]
    // loss = 0.41702986453303903
    auto output = make_tensor({1, 3}, {2.0f, 1.0f, 0.1f});
    auto target = make_tensor({1, 3}, {1.0f, 0.0f, 0.0f});
    Loss loss({output, target});

    REQUIRE(loss.cross_entropy() == Catch::Approx(0.41702986f).epsilon(1e-4));
}

TEST_CASE("cross_entropy assigns lower loss to a more confident correct prediction", "[loss][cross_entropy]") {
    // Expected values computed independently in Python from the same
    // formula as CrossEntropyOp::forward() (softmax then -log(p + 1e-7)).
    auto target = make_tensor({1, 3}, {1.0f, 0.0f, 0.0f});

    auto confident_output = make_tensor({1, 3}, {5.0f, 0.1f, 0.1f});
    auto unsure_output = make_tensor({1, 3}, {0.5f, 0.4f, 0.3f});

    Loss confident_loss({confident_output, target});
    Loss unsure_loss({unsure_output, target});

    REQUIRE(confident_loss.cross_entropy() == Catch::Approx(0.01478325f).epsilon(1e-4));
    REQUIRE(unsure_loss.cross_entropy() == Catch::Approx(1.00194258f).epsilon(1e-4));
    REQUIRE(confident_loss.cross_entropy() < unsure_loss.cross_entropy());
}