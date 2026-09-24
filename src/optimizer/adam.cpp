#include "optimizer/adam.hpp"

void Adam::step() {

    // Initialize m and v if they are empty
    if (m.empty()) {
        m.resize(params.size());
        v.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            m[i] = std::make_shared<Tensor>(params[i]->shape());
            v[i] = std::make_shared<Tensor>(params[i]->shape());
        }
    }

    // Bias correction: m and v start at zero, so after t steps they are
    // scaled down by (1 - beta^t). Dividing by it gives unbiased estimates
    // m_hat = m / (1 - beta1^t) and v_hat = v / (1 - beta2^t). Without it the
    // first updates are several times larger than learning_rate (about 3x at
    // t = 1 with the default betas), because v is biased towards zero much
    // longer than m. The factors are the same for every element, so they are
    // computed once per step.
    ++t;
    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));
    const float step_size = learning_rate / bias_correction1;
    const float sqrt_bias_correction2 = std::sqrt(bias_correction2);

    for (auto const &param : params) {
        auto &grad = param->get_grad();
        if (grad == nullptr) {
            throw std::runtime_error("Adam optimizer: gradient is null for a parameter");
        }

        size_t index = &param - &params[0]; // Get the index of the current parameter
        auto &m_t = m[index];
        auto &v_t = v[index];

        float *m_data = m_t->data();
        float *v_data = v_t->data();
        const float *grad_data = grad->data();
        float *param_data = param->data();
        const size_t size = param->size();

        for (size_t i = 0; i < size; ++i) {
            m_data[i] = beta1 * m_data[i] + (1 - beta1) * grad_data[i];
            v_data[i] = beta2 * v_data[i] + (1 - beta2) * grad_data[i] * grad_data[i];
        }

        // param -= lr * m_hat / (sqrt(v_hat) + epsilon), with the 1 / (1 - beta1^t)
        // of m_hat folded into step_size and sqrt(v_hat) = sqrt(v) / sqrt(1 - beta2^t).
        for (size_t i = 0; i < size; ++i) {
            param_data[i] -= step_size * m_data[i] / (std::sqrt(v_data[i]) / sqrt_bias_correction2 + epsilon);
        }
    }
}