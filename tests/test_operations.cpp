#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ops/average.hpp"
#include "ops/crossentropy.hpp"
#include "ops/matmul.hpp"
#include "ops/matsum.hpp"
#include "ops/relu.hpp"
#include "ops/softmax.hpp"

using Catch::Approx;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

static std::shared_ptr<Tensor> make_tensor(const std::vector<size_t> &shape, const std::vector<float> &values) {
    auto t = std::make_shared<Tensor>(shape);
    REQUIRE(t->size() == values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        t->set_data(i, values[i]);
    }
    return t;
}

// ---------------------------------------------------------------------------
// MatMulOp
// ---------------------------------------------------------------------------
TEST_CASE("MatMulOp: forward computes A @ B", "[operation][matmul][forward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({2, 2}, {5, 6, 7, 8});

    auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
    auto C = op->forward();

    // Expected: [[19, 22], [43, 50]]
    REQUIRE(C->data()[0] == Approx(19));
    REQUIRE(C->data()[1] == Approx(22));
    REQUIRE(C->data()[2] == Approx(43));
    REQUIRE(C->data()[3] == Approx(50));
}

TEST_CASE("MatMulOp: backward computes dA = dC @ B^T and dB = A^T @ dC", "[operation][matmul][backward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({2, 2}, {5, 6, 7, 8});

    auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
    op->forward();

    auto dC = make_tensor({2, 2}, {1, 2, 3, 4});
    op->backward(dC);

    // dA = dC @ B^T = [[17, 23], [39, 53]]
    REQUIRE(A->get_grad()->data()[0] == Approx(17));
    REQUIRE(A->get_grad()->data()[1] == Approx(23));
    REQUIRE(A->get_grad()->data()[2] == Approx(39));
    REQUIRE(A->get_grad()->data()[3] == Approx(53));

    // dB = A^T @ dC = [[10, 14], [14, 20]]
    REQUIRE(B->get_grad()->data()[0] == Approx(10));
    REQUIRE(B->get_grad()->data()[1] == Approx(14));
    REQUIRE(B->get_grad()->data()[2] == Approx(14));
    REQUIRE(B->get_grad()->data()[3] == Approx(20));
}

TEST_CASE("MatMulOp: forward() exceptions", "[operation][matmul][exceptions][forward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({2, 2}, {5, 6, 7, 8});

    SECTION("wrong number of inputs") {
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("null input") {
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A, nullptr});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("both operands below rank 2") {
        auto vecA = make_tensor({2}, {1, 2});
        auto vecB = make_tensor({2}, {3, 4});
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{vecA, vecB});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("incompatible inner dimensions") {
        auto C = make_tensor({3, 2}, {1, 2, 3, 4, 5, 6}); // (2x2) @ (3x2): 2 != 3
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A, C});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("incompatible batch dims for broadcasting") {
        auto A3 = make_tensor({2, 2, 2}, std::vector<float>(8, 1.0f));
        auto B3 = make_tensor({3, 2, 2}, std::vector<float>(12, 1.0f)); // batch: 2 vs 3
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A3, B3});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("zero-sized row/column") {
        auto Azero = std::make_shared<Tensor>(std::vector<size_t>{0, 2});
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{Azero, B});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
}

TEST_CASE("MatMulOp: backward() exceptions", "[operation][matmul][exceptions][backward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({2, 2}, {5, 6, 7, 8});
    auto dC = make_tensor({2, 2}, {1, 1, 1, 1});

    SECTION("null gradient") {
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
        op->forward();
        REQUIRE_THROWS_AS(op->backward(nullptr), std::invalid_argument);
    }
    SECTION("wrong number of inputs") {
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A});
        REQUIRE_THROWS_AS(op->backward(dC), std::invalid_argument);
    }
    SECTION("null input") {
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A, nullptr});
        REQUIRE_THROWS_AS(op->backward(dC), std::invalid_argument);
    }
    SECTION("both operands below rank 2") {
        auto vecA = make_tensor({2}, {1, 2});
        auto vecB = make_tensor({2}, {3, 4});
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{vecA, vecB});
        REQUIRE_THROWS_AS(op->backward(dC), std::invalid_argument);
    }
    SECTION("zero-sized gradient row/column") {
        auto zeroGrad = std::make_shared<Tensor>(std::vector<size_t>{0, 2});
        auto op = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
        REQUIRE_THROWS_AS(op->backward(zeroGrad), std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// MatSumOp
// ---------------------------------------------------------------------------
TEST_CASE("MatSumOp: forward computes A + B element-wise", "[operation][matsum][forward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({2, 2}, {10, 20, 30, 40});

    auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
    auto C = op->forward();

    REQUIRE(C->data()[0] == Approx(11));
    REQUIRE(C->data()[1] == Approx(22));
    REQUIRE(C->data()[2] == Approx(33));
    REQUIRE(C->data()[3] == Approx(44));
}

TEST_CASE("MatSumOp: backward with same-shape operands routes grad unchanged", "[operation][matsum][backward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({2, 2}, {10, 20, 30, 40});

    auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
    op->forward();

    auto dC = make_tensor({2, 2}, {1, 2, 3, 4});
    op->backward(dC);

    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(A->get_grad()->data()[i] == Approx(dC->data()[i]));
        REQUIRE(B->get_grad()->data()[i] == Approx(dC->data()[i]));
    }
}

TEST_CASE("MatSumOp: backward sums the gradient back over a broadcast axis", "[operation][matsum][backward][broadcast]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({1, 2}, {100, 200}); // broadcast over rows

    auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
    auto C = op->forward();

    REQUIRE(C->data()[0] == Approx(101));
    REQUIRE(C->data()[1] == Approx(202));
    REQUIRE(C->data()[2] == Approx(103));
    REQUIRE(C->data()[3] == Approx(204));

    auto dC = make_tensor({2, 2}, {1, 1, 1, 1});
    op->backward(dC);

    // dA: same shape as A, gradient passes through unchanged.
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(A->get_grad()->data()[i] == Approx(1));
    }
    // dB: the broadcast axis (rows) gets summed -> [1+1, 1+1] = [2, 2].
    REQUIRE(B->get_grad()->data()[0] == Approx(2));
    REQUIRE(B->get_grad()->data()[1] == Approx(2));
}

TEST_CASE("MatSumOp: forward() exceptions", "[operation][matsum][exceptions][forward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});

    SECTION("wrong number of inputs") {
        auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("null input") {
        auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, nullptr});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("incompatible shapes") {
        auto C = make_tensor({3, 3}, std::vector<float>(9, 1.0f));
        auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, C});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
}

TEST_CASE("MatSumOp: backward() exceptions", "[operation][matsum][exceptions][backward]") {
    auto A = make_tensor({2, 2}, {1, 2, 3, 4});
    auto B = make_tensor({2, 2}, {1, 1, 1, 1});
    auto dC = make_tensor({2, 2}, {1, 1, 1, 1});

    SECTION("null gradient") {
        auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
        op->forward();
        REQUIRE_THROWS_AS(op->backward(nullptr), std::invalid_argument);
    }
    SECTION("wrong number of inputs") {
        auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A});
        REQUIRE_THROWS_AS(op->backward(dC), std::invalid_argument);
    }
    SECTION("null input") {
        auto op = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, nullptr});
        REQUIRE_THROWS_AS(op->backward(dC), std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// ReluOp
// ---------------------------------------------------------------------------
TEST_CASE("ReluOp: forward computes max(0, x) element-wise", "[operation][relu][forward]") {
    auto X = make_tensor({2, 3}, {-2, -1, 0, 1, 2, 3});

    auto op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{X});
    auto Y = op->forward();

    REQUIRE(Y->data()[0] == Approx(0));
    REQUIRE(Y->data()[1] == Approx(0));
    REQUIRE(Y->data()[2] == Approx(0));
    REQUIRE(Y->data()[3] == Approx(1));
    REQUIRE(Y->data()[4] == Approx(2));
    REQUIRE(Y->data()[5] == Approx(3));
}

TEST_CASE("ReluOp: backward gates the gradient by the sign of the input", "[operation][relu][backward]") {
    auto X = make_tensor({2, 3}, {-2, -1, 0, 1, 2, 3});

    auto op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{X});
    op->forward();

    auto dY = make_tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    op->backward(dY);

    auto dX = X->get_grad();
    REQUIRE(dX->data()[0] == Approx(0)); // x = -2
    REQUIRE(dX->data()[1] == Approx(0)); // x = -1
    REQUIRE(dX->data()[2] == Approx(0)); // x =  0
    REQUIRE(dX->data()[3] == Approx(4)); // x =  1
    REQUIRE(dX->data()[4] == Approx(5)); // x =  2
    REQUIRE(dX->data()[5] == Approx(6)); // x =  3
}

TEST_CASE("ReluOp: exceptions", "[operation][relu][exceptions]") {
    SECTION("wrong number of inputs in forward") {
        auto X = make_tensor({2}, {1, 2});
        auto op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{X, X});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("null input in forward") {
        auto op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{nullptr});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("null gradient in backward") {
        auto X = make_tensor({2}, {1, 2});
        auto op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{X});
        op->forward();
        REQUIRE_THROWS_AS(op->backward(nullptr), std::invalid_argument);
    }
    SECTION("wrong number of inputs in backward") {
        auto X = make_tensor({2}, {1, 2});
        auto dY = make_tensor({2}, {1, 1});
        auto op = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{X, X});
        REQUIRE_THROWS_AS(op->backward(dY), std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// SoftmaxOp
// ---------------------------------------------------------------------------
TEST_CASE("SoftmaxOp: forward on a uniform row gives a uniform distribution", "[operation][softmax][forward]") {
    auto X = make_tensor({3}, {0, 0, 0});

    auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{X});
    auto Y = op->forward();

    REQUIRE(Y->data()[0] == Approx(1.0 / 3.0));
    REQUIRE(Y->data()[1] == Approx(1.0 / 3.0));
    REQUIRE(Y->data()[2] == Approx(1.0 / 3.0));
}

TEST_CASE("SoftmaxOp: forward matches the direct definition, row by row", "[operation][softmax][forward]") {
    // Two independent rows in a single (2,3) tensor, to also exercise the
    // "more than one row" path.
    std::vector<float> x = {1, 2, 3, 0, -1, 5};
    auto X = make_tensor({2, 3}, x);

    auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{X});
    auto Y = op->forward();

    for (size_t row = 0; row < 2; ++row) {
        size_t off = row * 3;
        float m = std::max({x[off], x[off + 1], x[off + 2]});
        float e0 = std::exp(x[off] - m);
        float e1 = std::exp(x[off + 1] - m);
        float e2 = std::exp(x[off + 2] - m);
        float sum = e0 + e1 + e2;

        REQUIRE(Y->data()[off] == Approx(e0 / sum));
        REQUIRE(Y->data()[off + 1] == Approx(e1 / sum));
        REQUIRE(Y->data()[off + 2] == Approx(e2 / sum));
        REQUIRE(Y->data()[off] + Y->data()[off + 1] + Y->data()[off + 2] == Approx(1.0f));
    }
}

TEST_CASE("SoftmaxOp: backward matches the softmax-Jacobian formula", "[operation][softmax][backward]") {
    std::vector<float> x = {1, 2, 3};
    auto X = make_tensor({3}, x);

    auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{X});
    auto Y = op->forward();

    std::vector<float> g = {1, 0, 0}; // fixed upstream gradient
    auto dY = make_tensor({3}, g);
    op->backward(dY);

    // Independent oracle, straight from the mathematical definition:
    // dx_j = y_j * (g_j - sum_k(g_k * y_k))
    float dot = 0.0f;
    for (size_t k = 0; k < 3; ++k)
        dot += g[k] * Y->data()[k];

    auto dX = X->get_grad();
    for (size_t j = 0; j < 3; ++j) {
        float expected = Y->data()[j] * (g[j] - dot);
        REQUIRE(dX->data()[j] == Approx(expected));
    }
}

TEST_CASE("SoftmaxOp: exceptions", "[operation][softmax][exceptions]") {
    SECTION("wrong number of inputs in forward") {
        auto X = make_tensor({3}, {1, 2, 3});
        auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{X, X});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("null input in forward") {
        auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{nullptr});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("backward called before forward") {
        auto X = make_tensor({3}, {1, 2, 3});
        auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{X});
        auto dY = make_tensor({3}, {1, 0, 0});
        REQUIRE_THROWS_AS(op->backward(dY), std::invalid_argument);
    }
    SECTION("null gradient in backward") {
        auto X = make_tensor({3}, {1, 2, 3});
        auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{X});
        op->forward();
        REQUIRE_THROWS_AS(op->backward(nullptr), std::invalid_argument);
    }
    SECTION("wrong number of inputs in backward") {
        auto X = make_tensor({3}, {1, 2, 3});
        auto dY = make_tensor({3}, {1, 0, 0});
        auto op = std::make_shared<SoftmaxOp>(std::vector<std::shared_ptr<Tensor>>{X, X});
        REQUIRE_THROWS_AS(op->backward(dY), std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// AverageOp
// ---------------------------------------------------------------------------

TEST_CASE("AverageOp: forward", "[operation][average][forward]") {
    SECTION("computes average of multiple elements") {
        auto X = make_tensor({4}, {1, 2, 3, 4});
        auto op = std::make_shared<AverageOp>(std::vector<std::shared_ptr<Tensor>>{X});
        auto Y = op->forward();
        REQUIRE(Y.size() == 1);
        REQUIRE(Y.data()[0] == Approx(2.5f));
    }

    SECTION("average of a single element equals the element itself") {
        auto X = make_tensor({1}, {7});
        auto op = std::make_shared<AverageOp>(std::vector<std::shared_ptr<Tensor>>{X});
        auto Y = op->forward();
        REQUIRE(Y.data()[0] == Approx(7.0f));
    }

    SECTION("handles negative values correctly") {
        auto X = make_tensor({3}, {-1, 0, 1});
        auto op = std::make_shared<AverageOp>(std::vector<std::shared_ptr<Tensor>>{X});
        auto Y = op->forward();
        REQUIRE(Y.data()[0] == Approx(0.0f));
    }
}

TEST_CASE("AverageOp: backward", "[operation][average][backward]") {
    SECTION("distributes gradient equally among inputs") {
        auto X = make_tensor({4}, {1, 2, 3, 4});
        auto op = std::make_shared<AverageOp>(std::vector<std::shared_ptr<Tensor>>{X});
        op->forward();

        auto dY = make_tensor({1}, {1});
        op->backward(dY);

        // Ogni elemento riceve grad / N
        auto const &grad_data = X->get_grad()->data();
        for (size_t i = 0; i < X->size(); ++i) {
            REQUIRE(grad_data[i] == Approx(0.25f));
        }
    }
}

// ---------------------------------------------------------------------------
// CrossEntropyOp
// ---------------------------------------------------------------------------
TEST_CASE("CrossEntropyOp: forward computes -sum(target * log(p)) per row", "[operation][crossentropy][forward]") {
    std::vector<float> p = {0.7f, 0.2f, 0.1f, 0.25f, 0.25f, 0.5f};
    std::vector<float> t = {1, 0, 0, 0, 0, 1};
    auto P = make_tensor({2, 3}, p);
    auto T = make_tensor({2, 3}, t);

    auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P, T});
    auto loss = op->forward();

    const float eps = 1e-7f;
    float expected0 = -std::log(p[0] + eps); // target picks column 0 on row 0
    float expected1 = -std::log(p[5] + eps); // target picks column 2 on row 1

    REQUIRE(loss->shape() == std::vector<size_t>{2, 1});
    REQUIRE(loss->data()[0] == Approx(expected0));
    REQUIRE(loss->data()[1] == Approx(expected1));
}

TEST_CASE("CrossEntropyOp: backward matches -grad_row * target / p", "[operation][crossentropy][backward]") {
    std::vector<float> p = {0.7f, 0.2f, 0.1f, 0.25f, 0.25f, 0.5f};
    std::vector<float> t = {1, 0, 0, 0, 0, 1};
    auto P = make_tensor({2, 3}, p);
    auto T = make_tensor({2, 3}, t);

    auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P, T});
    op->forward();

    auto dLoss = make_tensor({2, 1}, {1, 2}); // fixed upstream gradient, one per row
    op->backward(dLoss);

    const float eps = 1e-7f;
    auto dP = P->get_grad();
    for (size_t i = 0; i < 6; ++i) {
        size_t row = i / 3;
        float expected = -dLoss->data()[row] * t[i] / (p[i] + eps);
        REQUIRE(dP->data()[i] == Approx(expected));
    }
}

TEST_CASE("CrossEntropyOp: forward() exceptions", "[operation][crossentropy][exceptions][forward]") {
    auto P = make_tensor({2, 3}, {0.7f, 0.2f, 0.1f, 0.25f, 0.25f, 0.5f});
    auto T = make_tensor({2, 3}, {1, 0, 0, 0, 0, 1});

    SECTION("wrong number of inputs") {
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("null input") {
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P, nullptr});

        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("mismatched shapes") {
        auto T2 = make_tensor({3, 2}, {1, 0, 0, 0, 0, 1});
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P, T2});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
    SECTION("zero-length last dimension") {
        auto Pzero = std::make_shared<Tensor>(std::vector<size_t>{2, 0});
        auto Tzero = std::make_shared<Tensor>(std::vector<size_t>{2, 0});
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{Pzero, Tzero});
        REQUIRE_THROWS_AS(op->forward(), std::invalid_argument);
    }
}

TEST_CASE("CrossEntropyOp: backward() exceptions", "[operation][crossentropy][exceptions][backward]") {
    auto P = make_tensor({2, 3}, {0.7f, 0.2f, 0.1f, 0.25f, 0.25f, 0.5f});
    auto T = make_tensor({2, 3}, {1, 0, 0, 0, 0, 1});
    auto dLoss = make_tensor({2, 1}, {1, 1});

    SECTION("null gradient") {
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P, T});
        op->forward();
        REQUIRE_THROWS_AS(op->backward(nullptr), std::invalid_argument);
    }
    SECTION("wrong number of inputs") {
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P});
        REQUIRE_THROWS_AS(op->backward(dLoss), std::invalid_argument);
    }
    SECTION("null input") {
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P, nullptr});
        REQUIRE_THROWS_AS(op->backward(dLoss), std::invalid_argument);
    }
    SECTION("mismatched shapes") {
        auto T2 = make_tensor({3, 2}, {1, 0, 0, 0, 0, 1});
        auto op = std::make_shared<CrossEntropyOp>(std::vector<std::shared_ptr<Tensor>>{P, T2});
        REQUIRE_THROWS_AS(op->backward(dLoss), std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// Chaining two operations: MatSumOp -> ReluOp
// ---------------------------------------------------------------------------
// Checks that the pointer saved by set_operation() / o_inputs is enough to
// manually chain two backward() calls and get correct gradients.
TEST_CASE("Chaining MatSumOp -> ReluOp propagates gradients through both ops", "[operation][chain][matsum][relu]") {
    auto A = make_tensor({2, 2}, {1, -5, 3, 4});
    auto B = make_tensor({2, 2}, {1, 1, 1, 1});

    auto sumOp = std::make_shared<MatSumOp>(std::vector<std::shared_ptr<Tensor>>{A, B});
    auto S = sumOp->forward(); // S = A + B = [[2,-4],[4,5]]

    auto reluOp = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{S});
    auto Y = reluOp->forward(); // relu(S) = [[2,0],[4,5]]

    REQUIRE(Y->data()[0] == Approx(2));
    REQUIRE(Y->data()[1] == Approx(0));
    REQUIRE(Y->data()[2] == Approx(4));
    REQUIRE(Y->data()[3] == Approx(5));

    auto dY = make_tensor({2, 2}, {1, 2, 3, 4});
    reluOp->backward(dY); // populates S->get_grad()

    REQUIRE(S->get_grad() != nullptr);
    REQUIRE(S->get_grad()->data()[0] == Approx(1)); // S[0] =  2 > 0
    REQUIRE(S->get_grad()->data()[1] == Approx(0)); // S[1] = -4 <= 0
    REQUIRE(S->get_grad()->data()[2] == Approx(3)); // S[2] =  4 > 0
    REQUIRE(S->get_grad()->data()[3] == Approx(4)); // S[3] =  5 > 0

    sumOp->backward(S->get_grad()); // propagates further back to A and B

    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(A->get_grad()->data()[i] == Approx(S->get_grad()->data()[i]));
        REQUIRE(B->get_grad()->data()[i] == Approx(S->get_grad()->data()[i]));
    }
}

// ---------------------------------------------------------------------------
// A single layer: y = relu(W @ x)
// ---------------------------------------------------------------------------
TEST_CASE("A single layer (MatMulOp -> ReluOp) forward and backward", "[operation][layer][matmul][relu]") {

    // a layer with 3 inputs and two outputs
    auto W = make_tensor({2, 3}, {1, -1, 2, 0, 1, -1});
    auto x = make_tensor({3, 1}, {1, 2, 3});

    auto matmulOp = std::make_shared<MatMulOp>(std::vector<std::shared_ptr<Tensor>>{W, x});
    auto Z = matmulOp->forward(); // W @ x = [[5],[-1]]

    REQUIRE(Z->data()[0] == Approx(5));
    REQUIRE(Z->data()[1] == Approx(-1));

    auto reluOp = std::make_shared<ReluOp>(std::vector<std::shared_ptr<Tensor>>{Z});
    auto Y = reluOp->forward(); // relu(Z) = [[5],[0]]

    REQUIRE(Y->data()[0] == Approx(5));
    REQUIRE(Y->data()[1] == Approx(0));

    auto dY = make_tensor({2, 1}, {1, 2});
    reluOp->backward(dY); // -> Z->get_grad() = [[1],[0]]

    REQUIRE(Z->get_grad()->data()[0] == Approx(1));
    REQUIRE(Z->get_grad()->data()[1] == Approx(0));

    matmulOp->backward(Z->get_grad()); // -> dW, dx

    // dW = dz @ x^T = [[1,2,3],[0,0,0]]
    REQUIRE(W->get_grad()->data()[0] == Approx(1));
    REQUIRE(W->get_grad()->data()[1] == Approx(2));
    REQUIRE(W->get_grad()->data()[2] == Approx(3));
    REQUIRE(W->get_grad()->data()[3] == Approx(0));
    REQUIRE(W->get_grad()->data()[4] == Approx(0));
    REQUIRE(W->get_grad()->data()[5] == Approx(0));

    // dx = W^T @ dz = [1, -1, 2]
    REQUIRE(x->get_grad()->data()[0] == Approx(1));
    REQUIRE(x->get_grad()->data()[1] == Approx(-1));
    REQUIRE(x->get_grad()->data()[2] == Approx(2));
}