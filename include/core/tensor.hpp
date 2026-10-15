#pragma once

#include "core/device.hpp"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using std::vector;

class Tensor;
class Operation; // Forward declaration di Operation

using OpToOutputMap = std::unordered_map<std::shared_ptr<Operation>, std::shared_ptr<Tensor>>;

/**
 * @brief A dense n-dimensional array with autograd support.
 *
 * Tensor owns its shape/strides by value (deep copy on every copy),
 * but its float buffer is a shared_ptr<float[]>: (shallow copy on every copy),
 * transpose() and permute() share the same buffer, while
 * clone() allocates a fully independent buffer.
 *
 * The buffer lives on device(): in RAM for Device::CPU, in the GPU memory
 * (VRAM) for Device::CUDA. Shape, strides and all the other members always
 * live in RAM. A CUDA buffer is freed with cuda_free() by the shared_ptr
 * deleter, when the last tensor sharing it is destroyed.
 */
class Tensor : public std::enable_shared_from_this<Tensor> {
  public:
    // ---------- Construction ----------

    /**
     * @brief Allocates a new zero-initialized tensor with the given shape.
     * @param shape Size of each dimension.
     * @param require_grad Whether this tensor should track gradients.
     * @param device Where the buffer is allocated.
     */
    explicit Tensor(vector<size_t> shape, bool require_grad = true, Device device = Device::CPU);

    /**
     * @brief Builds a tensor from an existing buffer in CPU memory.
     *
     * @param shape Size of each dimension.
     * @param data Buffer in CPU memory holding at least the product of
     *        @p shape floats, or nullptr:
     *        - Device::CPU: wrapped without copying, so the tensor shares
     *          ownership of @p data with the caller (shallow copy);
     *        - Device::CUDA: copied into a new GPU buffer (not shared);
     *        - nullptr: a new zero-initialized buffer is allocated on @p device.
     * @param require_grad Whether this tensor should track gradients.
     * @param device Where the tensor's buffer lives.
     */
    Tensor(vector<size_t> shape, std::shared_ptr<float[]> data, bool require_grad = true, Device device = Device::CPU);

    // ---------- Device ----------

    /// @brief Returns where the buffer lives (CPU memory or GPU memory).
    Device device() const;

    /**
     * @brief Moves this tensor (in place) to @p target, together with its
     *        gradient if it has one. Does nothing if it is already there.
     *
     * The whole buffer is copied as-is and the strides are kept, so a
     * non-contiguous view is read in the same way after the move. The tensor
     * gets a new buffer: other tensors that shared the old one (views,
     * copies) keep it and stay on the old device.
     *
     * @param target Destination device.
     */
    void to(Device target);

    // ---------- Shape and views ----------

    /**
     * @brief Changes the logical shape without moving any data.
     * @param new_shape Must have the same total element count as the
     *        current shape.
     * @throws std::invalid_argument If the element counts don't match.
     */
    void reshape(vector<size_t> new_shape);

    /**
     * @brief Returns a view with reversed axis order. Shares the same data buffer;
     *        is detached from the autograd graph (no gradient, no producing
     *        operation), so gradients do not flow back through it.
     * @throws std::invalid_argument If the element counts don't match.
     */
    Tensor transpose() const;

    /**
     * @brief Returns a view with axis reordered according to @p new_axis.
     *        Shares the same data buffer; is detached from the autograd graph
     *        (no gradient, no producing operation), like transpose().
     * @param new_axis Permutation of [0, rank): new_axis[i] gives the
     *        original axis that becomes axis i in the result.
     * @throws std::invalid_argument If new_axis.size() != shape().size().
     * @throws std::invalid_argument If new_axis[i] == new_axis[j].
     * @throws std::out_of_bound If new_axis[i] exceed the shape().size().
     */
    Tensor permute(const vector<size_t> &new_axis) const;

    /**
     * @brief True if the buffer is laid out in row-major order with no gaps,
     *        i.e. element i of the logical order is data()[i]. Always true for
     *        a newly created tensor; false for most views (transpose()/permute()).
     */
    bool is_contiguous() const;

    /**
     * @brief Returns a deep copy with its own independent, contiguous data
     *        buffer holding the same logical values (a non-contiguous view is
     *        materialized in row-major order), on the same device. The
     *        gradient is not copied.
     * @throws std::runtime_error If the tensor is a non-contiguous view on
     *         the GPU (not supported yet).
     */
    Tensor clone() const;

    // ---------- Autograd ----------

    /**
     * @brief Backpropagates from this tensor through the whole computational
     *        graph that produced it.
     *
     * The gradient of this tensor must already be set (e.g. to 1 for a scalar
     * loss). The graph is visited in topological order, from this tensor back
     * to the leaves, and each operation's backward() is called once with the
     * gradient of its output. Gradients are accumulated (+=) into the inputs'
     * gradients, so they must be reset between two training steps.
     *
     * @throws std::runtime_error If this tensor does not require gradients.
     * @throws std::runtime_error If this tensor has no gradient set.
     */
    void backward();

    bool requires_grad() const;
    void set_requires_grad(bool req);

    /// @brief Returns the gradient (nullptr if it was never set).
    const std::shared_ptr<Tensor> &get_grad() const;
    void set_grad(std::shared_ptr<Tensor> new_grad);

    /// @brief Returns the operation that produced this tensor (nullptr for a leaf).
    const std::shared_ptr<Operation> &get_operation() const;
    void set_operation(std::shared_ptr<Operation> operation);

    // ---------- Data access ----------

    const vector<size_t> &shape() const;
    const vector<size_t> &strides() const;

    /// @brief Total number of elements (product of shape()).
    size_t size() const;

    /**
     * @brief Returns the raw pointer to the underlying buffer.
     *
     * Defined inline so that element access in hot loops compiles to a plain
     * load/store. The pointer stays valid as long as this tensor (or any
     * other tensor sharing the same buffer) is alive. Index it with strides()
     * when the tensor is a non-contiguous view (transpose()/permute()).
     *
     * @warning On Device::CUDA it is a GPU address: pass it to kernels and to
     *          the cuda_* functions, never dereference it on the CPU.
     */
    float *data() const { return t_data.get(); }

    /**
     * @brief Returns the only element of the tensor. On the GPU the value is
     *        first copied to the CPU.
     * @throws std::invalid_argument If the tensor does not have exactly one element.
     */
    float item() const;

  private:
    vector<size_t> t_shape;
    std::shared_ptr<float[]> t_data;
    vector<size_t> t_strides;
    size_t totalSize;
    std::shared_ptr<Tensor> grad = nullptr;
    bool t_require_grad;
    std::shared_ptr<Operation> t_operation = nullptr;
    Device t_device = Device::CPU;

    /// @brief Compute the strides based on the @p shape.
    void computeStrides();

    /// @brief Product of all dimensions in @p shape.
    static size_t computeTotalSize(const vector<size_t> &shape);

    /// @brief Allocates a new zero-initialized buffer of totalSize floats on t_device.
    void allocate_zeroed_buffer();

    void createComputationalgraph(std::vector<std::shared_ptr<Operation>> &computational_graph, std::shared_ptr<Tensor> tensor,
                                  OpToOutputMap &op_to_output);
};
