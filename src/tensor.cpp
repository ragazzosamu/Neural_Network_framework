#include "tensor.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>

using std::vector;

Tensor::Tensor(vector<size_t> shape, bool require_grad)
    : t_shape(std::move(shape)), t_require_grad(require_grad) {
    totalSize = computeTotalSize(t_shape);
    computeStrides();
    t_data = std::shared_ptr<float[]>(new float[totalSize]);
}

Tensor::Tensor(vector<size_t> shape, std::shared_ptr<float[]> data, bool require_grad)
    : t_shape(std::move(shape)), t_require_grad(require_grad) {
    totalSize = computeTotalSize(t_shape);
    computeStrides();
    if (data == nullptr) {
        t_data = std::shared_ptr<float[]>(new float[totalSize]);
    } else {
        t_data = std::move(data);
    }
}

// ---------- Operation on the shape ----------

void Tensor::reshape(vector<size_t> new_shape) {
    size_t new_size = computeTotalSize(new_shape);
    if (new_size != totalSize) {
        throw std::invalid_argument(
            "Reshape not valid: the total number of elements doesn't coincide");
    }
    t_shape = std::move(new_shape);
    computeStrides();
}

Tensor Tensor::transpose() const {
    Tensor transposed = *this;
    // A new autograd node: the old gradient no longer applies to this
    // (differently-shaped) view.
    transposed.grad = nullptr;
    std::reverse(transposed.t_shape.begin(), transposed.t_shape.end());
    std::reverse(transposed.t_strides.begin(), transposed.t_strides.end());
    return transposed;
}

Tensor Tensor::permute(const vector<size_t> &new_axis) const {
    if (new_axis.size() != t_shape.size()) {
        throw std::invalid_argument("Number of axis not valid: the total number of axis must "
                                    "be the same as the total number of the original shape");
    }

    std::vector<bool> seen(t_shape.size(), false);
    for (size_t i = 0; i < new_axis.size(); ++i) {
        size_t axis = new_axis[i];

        if (axis >= t_shape.size()) {
            throw std::out_of_range("Axis at position " + std::to_string(i) + " (" +
                                    std::to_string(axis) + ") exceeds max dimension (" +
                                    std::to_string(t_shape.size() - 1) + ")");
        }

        if (seen[axis]) {
            throw std::invalid_argument("Duplicate axis " + std::to_string(axis) + " at index " +
                                        std::to_string(i));
        }

        seen[axis] = true;
    }
    Tensor permutated = *this;
    // A new autograd node: the old gradient no longer applies to this
    // (differently-shaped) view.
    permutated.grad = nullptr;
    for (size_t i = 0; i < t_shape.size(); ++i) {
        permutated.t_shape[i] = t_shape[new_axis[i]];
        permutated.t_strides[i] = t_strides[new_axis[i]];
    }
    return permutated;
}

Tensor Tensor::clone() const {
    auto new_data = std::shared_ptr<float[]>(new float[totalSize]);
    std::copy(t_data.get(), t_data.get() + totalSize, new_data.get());

    Tensor cloned(t_shape, new_data);
    cloned.set_requires_grad(t_require_grad);
    return cloned;
}

// ---------- Accessor ----------

bool Tensor::requires_grad() const { return t_require_grad; }
void Tensor::set_requires_grad(bool req) { t_require_grad = req; }

const vector<size_t> &Tensor::shape() const { return t_shape; }
const vector<size_t> &Tensor::strides() const { return t_strides; }
size_t Tensor::size() const { return totalSize; }
std::shared_ptr<float[]> Tensor::data() const { return t_data; }

std::shared_ptr<Tensor> Tensor::get_grad() const { return grad; }
void Tensor::set_grad(std::shared_ptr<Tensor> new_grad) { grad = std::move(new_grad); }

// ---------- Private Helper ----------

void Tensor::computeStrides() {
    // Row-major layout: the last dimension is 1 (jump from one column to another),
    // the i stride is the product of all dimension sizes to its right
    // Example: shape = {2, 3, 4} -> strides = {12, 4, 1}

    t_strides.assign(t_shape.size(), 1);
    if (!t_shape.empty()) {
        for (int i = static_cast<int>(t_shape.size()) - 2; i >= 0; --i) {
            t_strides[i] = t_strides[i + 1] * t_shape[i + 1];
        }
    }
}

size_t Tensor::computeTotalSize(const vector<size_t> &shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    return size;
}
