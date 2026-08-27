#include "tensor.hpp"
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <memory>

TEST_CASE("Initialization test", "[Tensor][Initialization]") {
    vector<size_t> shape{4, 3};
    std::shared_ptr<float[]> data(
        new float[]{0.5f, 0.7f, 0.4f, 0.5f, 0.7f, 0.4f, 0.5f, 0.7f, 0.4f, 0.5f, 0.7f, 0.4f});

    SECTION("Without data") {
        Tensor tensor(shape);
        REQUIRE(tensor.size() == 12);
    }

    SECTION("With data") {
        Tensor tensor(shape, data);
        REQUIRE(tensor.size() == 12);
        REQUIRE(tensor.data() == data);
    }

    SECTION("Requires grad flag") {
        Tensor tensor(shape, false);
        REQUIRE(tensor.requires_grad() == false);
        tensor.set_requires_grad(true);
        REQUIRE(tensor.requires_grad() == true);
    }

    SECTION("Empty shape") {
        Tensor tensor(vector<size_t>{});
        REQUIRE(tensor.size() == 1);
        REQUIRE(tensor.shape().empty());
        REQUIRE(tensor.strides().empty());
    }

    SECTION("Zero dimension") {
        Tensor tensor(vector<size_t>{0, 3});
        REQUIRE(tensor.size() == 0);
    }
}

TEST_CASE("Transpose", "[Tensor][Transpose]") {
    SECTION("2D") {
        vector<size_t> shape{4, 3};
        Tensor tensor(shape);

        Tensor transposed = tensor.transpose();
        REQUIRE(transposed.shape()[0] == 3);
        REQUIRE(transposed.shape()[1] == 4);

        REQUIRE(transposed.data() == tensor.data());

        REQUIRE(transposed.strides()[0] == 1);
        REQUIRE(transposed.strides()[1] == 3);

        REQUIRE(tensor.shape()[0] == 4);
        REQUIRE(tensor.shape()[1] == 3);
        REQUIRE(tensor.strides()[0] == 3);
        REQUIRE(tensor.strides()[1] == 1);
    }

    SECTION("3D") {
        vector<size_t> shape{4, 3, 2};
        Tensor tensor(shape);

        Tensor transposed = tensor.transpose();
        REQUIRE(transposed.shape()[0] == 2);
        REQUIRE(transposed.shape()[1] == 3);
        REQUIRE(transposed.shape()[2] == 4);

        REQUIRE(transposed.strides()[0] == 1);
        REQUIRE(transposed.strides()[1] == 2);
        REQUIRE(transposed.strides()[2] == 6);
    }
}

TEST_CASE("Permute", "[Tensor][Permute]") {
    vector<size_t> shape{4, 3, 2};
    Tensor tensor(shape);

    SECTION("Right permutation") {
        vector<size_t> new_axis{2, 0, 1};
        Tensor permutated = tensor.permute(new_axis);

        REQUIRE(permutated.shape()[0] == tensor.shape()[2]);
        REQUIRE(permutated.shape()[1] == tensor.shape()[0]);
        REQUIRE(permutated.shape()[2] == tensor.shape()[1]);

        REQUIRE(permutated.strides()[0] == 1);
        REQUIRE(permutated.strides()[1] == 6);
        REQUIRE(permutated.strides()[2] == 2);

        REQUIRE(permutated.data() == tensor.data());

        REQUIRE(tensor.shape()[0] == 4);
        REQUIRE(tensor.shape()[1] == 3);
        REQUIRE(tensor.shape()[2] == 2);
        REQUIRE(tensor.strides()[0] == 6);
        REQUIRE(tensor.strides()[1] == 2);
        REQUIRE(tensor.strides()[2] == 1);
    }

    SECTION("Identity permutation") {
        vector<size_t> new_axis{0, 1, 2};
        Tensor permutated = tensor.permute(new_axis);

        REQUIRE(permutated.shape()[0] == tensor.shape()[0]);
        REQUIRE(permutated.shape()[1] == tensor.shape()[1]);
        REQUIRE(permutated.shape()[2] == tensor.shape()[2]);

        REQUIRE(permutated.strides()[0] == tensor.strides()[0]);
        REQUIRE(permutated.strides()[1] == tensor.strides()[1]);
        REQUIRE(permutated.strides()[2] == tensor.strides()[2]);
    }

    SECTION("Index i out of bound") {
        vector<size_t> new_axis{4, 0, 1};
        REQUIRE_THROWS_AS(tensor.permute(new_axis), std::out_of_range);
    }

    SECTION("new_axis' wrong dimension") {
        vector<size_t> new_axis{0, 1};
        REQUIRE_THROWS_AS(tensor.permute(new_axis), std::invalid_argument);
    }

    SECTION("Duplicated axis") {
        vector<size_t> new_axis{0, 1, 1};
        REQUIRE_THROWS_AS(tensor.permute(new_axis), std::invalid_argument);
    }
}

TEST_CASE("Reshape", "[Tensor][Reshape]") {
    vector<size_t> shape{4, 3, 2};
    Tensor tensor(shape);

    SECTION("Right Reshape") {
        vector<size_t> new_shape{2, 12};
        tensor.reshape(new_shape);
        REQUIRE(tensor.shape()[0] == 2);
        REQUIRE(tensor.shape()[1] == 12);
        REQUIRE(tensor.size() == 24);

        REQUIRE(tensor.strides()[0] == 12);
        REQUIRE(tensor.strides()[1] == 1);
    }

    SECTION("Reshape to 1D") {
        vector<size_t> new_shape{24};
        tensor.reshape(new_shape);
        REQUIRE(tensor.shape()[0] == 24);
        REQUIRE(tensor.strides()[0] == 1);
        REQUIRE(tensor.size() == 24);
    }

    SECTION("Reshape no-op") {
        vector<size_t> new_shape{4, 3, 2};
        tensor.reshape(new_shape);
        REQUIRE(tensor.shape()[0] == 4);
        REQUIRE(tensor.shape()[1] == 3);
        REQUIRE(tensor.shape()[2] == 2);
        REQUIRE(tensor.strides()[0] == 6);
        REQUIRE(tensor.strides()[1] == 2);
        REQUIRE(tensor.strides()[2] == 1);
    }

    SECTION("Wrong Reshape") {
        vector<size_t> new_shape{1, 12};
        REQUIRE_THROWS_AS(tensor.reshape(new_shape), std::invalid_argument);
    }
}

TEST_CASE("Clone", "[Tensor][Clone]") {
    vector<size_t> shape{4, 3, 2};
    std::shared_ptr<float[]> data(new float[24]);
    for (size_t i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i);
    }
    Tensor tensor(shape, data);
    Tensor cloned = tensor.clone();

    REQUIRE(cloned.shape()[0] == tensor.shape()[0]);
    REQUIRE(cloned.shape()[1] == tensor.shape()[1]);
    REQUIRE(cloned.shape()[2] == tensor.shape()[2]);

    REQUIRE(cloned.data() != tensor.data());

    bool same_values = true;
    for (size_t i = 0; i < tensor.size() && same_values; ++i) {
        if (cloned.data()[i] != tensor.data()[i]) {
            same_values = false;
        }
    }
    REQUIRE(same_values);

    cloned.data()[0] = 999.0f;
    REQUIRE(tensor.data()[0] != 999.0f);

    REQUIRE(cloned.requires_grad() == tensor.requires_grad());
}

TEST_CASE("Grad accessors", "[Tensor][Grad]") {
    vector<size_t> shape{2, 2};
    Tensor tensor(shape);

    REQUIRE(tensor.get_grad() == nullptr);

    auto grad_tensor = std::shared_ptr<Tensor>(new Tensor(shape));
    tensor.set_grad(grad_tensor);

    REQUIRE(tensor.get_grad() == grad_tensor);
    REQUIRE(tensor.get_grad()->shape()[0] == 2);
}