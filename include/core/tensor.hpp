#pragma once

#include "ops/operation.hpp"
#include <cstddef>
#include <memory>
#include <vector>
using std::vector;

class Operation; // Forward declaration di Operation
/**
 * @brief A dense n-dimensional array with autograd support.
 *
 * Tensor owns its shape/strides by value (deep copy on every copy),
 * but its float buffer is a shared_ptr<float[]>: (shallow copy on every copy),
 * transpose() and permute() share the same buffer, while
 * clone() allocates a fully independent buffer.
 */
class Tensor {
  public:
    /**
     * @brief Allocates a new zero-initialized tensor with the given shape.
     * @param shape Size of each dimension.
     * @param require_grad Whether this tensor should track gradients.
     */
    explicit Tensor(vector<size_t> shape, bool require_grad = true);

    /**
     * @brief Wraps an existing buffer without copying its contents.
     *
     * The resulting tensor shares ownership of @p data with the caller
     * (shallow copy). If @p data is null, a new zero-initialized
     * buffer is allocated instead.
     *
     * @param shape Size of each dimension.
     * @param data Buffer to wrap.
     * @param require_grad Whether this tensor should track gradients.
     */
    Tensor(vector<size_t> shape, std::shared_ptr<float[]> data, bool require_grad = true);

    /**
     * @brief Changes the logical shape without moving any data.
     * @param new_shape Must have the same total element count as the
     *        current shape.
     * @throws std::invalid_argument If the element counts don't match.
     */
    void reshape(vector<size_t> new_shape);

    /**
     * @brief Returns a view with reversed axis order. Shares the same data buffer;
     *        does not carry the gradients
     * @throws std::invalid_argument If the element counts don't match.
     */
    Tensor transpose() const;

    /**
     * @brief Returns a view with axis reordered according to @p new_axis.
     *        Shares the same data buffer; does not carry the gradients
     * @param new_axis Permutation of [0, rank): new_axis[i] gives the
     *        original axis that becomes axis i in the result.
     * @throws std::invalid_argument If new_axis.size() != shape().size().
     * @throws std::invalid_argument If new_axis[i] == new_axis[j].
     * @throws std::out_of_bound If new_axis[i] exceed the shape().size().
     */
    Tensor permute(const vector<size_t> &new_axis) const;

    /**
     * @brief Returns a deep copy with its own independent data buffer.
     *        The gradient is not copied.
     */
    Tensor clone() const;

    bool requires_grad() const;
    void set_requires_grad(bool req);

    const vector<size_t> &shape() const;
    const vector<size_t> &strides() const;
    size_t size() const;
    std::shared_ptr<float[]> data() const;
    void set_data(size_t i, float value);
    void add_to_data(size_t i, float value);

    std::shared_ptr<Tensor> get_grad() const;
    void set_grad(std::shared_ptr<Tensor> new_grad);

    void set_operation(std::shared_ptr<Operation> operation);

  private:
    vector<size_t> t_shape;
    std::shared_ptr<float[]> t_data;
    vector<size_t> t_strides;
    size_t totalSize;
    std::shared_ptr<Tensor> grad = nullptr;
    bool t_require_grad;
    std::shared_ptr<Operation> t_operation = nullptr;

    /// @brief Compute the strides based on the @p shape.
    void computeStrides();

    /// @brief Product of all dimensions in @p shape.
    static size_t computeTotalSize(const vector<size_t> &shape);
};