#include "core/tensor.hpp"
#include "ops/operation.hpp"

using std::vector;

Tensor::Tensor(vector<size_t> shape, bool require_grad) : t_shape(std::move(shape)), t_require_grad(require_grad) {
    totalSize = computeTotalSize(t_shape);
    computeStrides();
    t_data = std::make_shared<float[]>(totalSize);
}

Tensor::Tensor(vector<size_t> shape, std::shared_ptr<float[]> data, bool require_grad) : t_shape(std::move(shape)), t_require_grad(require_grad) {
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
        throw std::invalid_argument("Reshape not valid: the total number of elements doesn't coincide");
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
            throw std::out_of_range("Axis at position " + std::to_string(i) + " (" + std::to_string(axis) + ") exceeds max dimension (" +
                                    std::to_string(t_shape.size() - 1) + ")");
        }

        if (seen[axis]) {
            throw std::invalid_argument("Duplicate axis " + std::to_string(axis) + " at index " + std::to_string(i));
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

// ---------- Backward ----------
void Tensor::backward() {
    if (!t_require_grad) {
        throw std::runtime_error("Cannot call backward() on a tensor that does not require gradients");
    }

    if (grad == nullptr) {
        throw std::runtime_error("Cannot call backward() on a tensor with no gradient");
    }

    std::vector<std::shared_ptr<Operation>> computational_graph_list;
    // map each operation with its output
    //  use this map both for visited checking and to find the right gradient for the backward
    OpToOutputMap op_to_output;

    createComputationalgraph(computational_graph_list, shared_from_this(), op_to_output);

    for (auto it = computational_graph_list.rbegin(); it != computational_graph_list.rend(); ++it) {
        auto op = *it;
        auto output_tensor = op_to_output[op];
        op->backward(output_tensor->get_grad());
    }
}

// ---------- Accessor ----------

bool Tensor::requires_grad() const { return t_require_grad; }
void Tensor::set_requires_grad(bool req) { t_require_grad = req; }

const vector<size_t> &Tensor::shape() const { return t_shape; }
const vector<size_t> &Tensor::strides() const { return t_strides; }
size_t Tensor::size() const { return totalSize; }

std::shared_ptr<float[]> Tensor::data() const { return t_data; }
void Tensor::set_data(size_t i, float value) { t_data[i] = value; }
void Tensor::add_to_data(size_t i, float value) { t_data[i] += value; }

const std::shared_ptr<Tensor> &Tensor::get_grad() const { return grad; }
void Tensor::set_grad(std::shared_ptr<Tensor> new_grad) { grad = std::move(new_grad); }

const std::shared_ptr<Operation> &Tensor::get_operation() const { return t_operation; }
void Tensor::set_operation(std::shared_ptr<Operation> operation) { t_operation = std::move(operation); }

float Tensor::item() const {
    if (totalSize != 1) {
        throw std::invalid_argument("Tensor must have exactly one element to call item()");
    }
    return t_data[0];
}

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

void Tensor::createComputationalgraph(std::vector<std::shared_ptr<Operation>> &computational_graph, std::shared_ptr<Tensor> tensor,
                                      OpToOutputMap &op_to_output) {

    if (!tensor)
        return;
    auto op = tensor->get_operation();

    if (!op || op_to_output.find(op) != op_to_output.end())
        return;

    auto &op_inputs = op->inputs();
    op_to_output[op] = tensor;
    // create the graph recursively
    // First append the inputs and then the operation, in this way every output has its inputs on the left
    for (auto &input : op_inputs) {
        createComputationalgraph(computational_graph, input, op_to_output);
    }
    // Appended to the end because it is more efficient than inserting at the front
    computational_graph.push_back(op);
}