#include "optimizer.hpp"
#include <cmath>
#include <stdexcept>

/**
 * @brief Stochastic gradient descent optimizer.
 *
 * SGD updates each parameter by subtracting the learning rate times the
 * corresponding gradient.
 */
class SGD : public Optimizer {

  public:
    /**
     * @brief Constructs an SGD optimizer.
     *
     * @param learning_rate Step size applied to each gradient descent update.
     * @param params Trainable parameters to optimize.
     */
    explicit SGD(float learning_rate, std::vector<std::shared_ptr<Tensor>> params) : Optimizer(learning_rate, std::move(params)) {}

    /**
     * @brief Performs one SGD update step.
     *
     * @throws std::runtime_error if any parameter's gradient is null.
     */
    void step() override;
};