#include "optimizer.hpp"
#include <cmath>
#include <stdexcept>

/**
 * @brief Adam optimizer with adaptive moment estimation.
 *
 * Adam combines first-order and second-order moment estimates to scale each
 * parameter update based on the average gradient and its squared value.
 *
 * Both moment estimates start at zero, so during the first steps they are
 * biased towards zero (the second one much longer, since beta2 is closer to 1).
 * As in the original algorithm (Kingma & Ba, 2015), they are bias-corrected
 * by dividing them by (1 - beta^t), where t is the step count.
 */
class Adam : public Optimizer {

  public:
    /**
     * @brief Constructs an Adam optimizer.
     *
     * @param learning_rate Step size used by the optimizer.
     * @param params Trainable parameters to optimize.
     * @param beta Exponential decay rate for the first moment estimate.
     * @param beta2 Exponential decay rate for the second moment estimate.
     * @param epsilon Small constant added for numerical stability.
     */
    Adam(float learning_rate, std::vector<std::shared_ptr<Tensor>> params, float beta = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
        : Optimizer(learning_rate, std::move(params)), beta1(beta), beta2(beta2), epsilon(epsilon) {}

    /**
     * @brief Performs one Adam update step.
     *
     * @throws std::runtime_error if any parameter's gradient is null.
     */
    void step() override;

  private:
    /// Exponential decay rate for the first moment estimate.
    float beta1;

    /// Exponential decay rate for the second moment estimate.
    float beta2;

    /// Numerical stability constant used to avoid division by zero.
    float epsilon;

    /// First moment estimate for each parameter.
    std::vector<std::shared_ptr<Tensor>> m;

    /// Second moment estimate for each parameter.
    std::vector<std::shared_ptr<Tensor>> v;

    /// Number of step() calls so far (t in the bias correction).
    size_t t = 0;
};