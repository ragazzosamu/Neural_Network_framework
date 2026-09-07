#include "optimizer/Adam.hpp"

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
    for (auto const &param : params) {
        auto &grad = param->get_grad();
        if (grad == nullptr) {
            throw std::runtime_error("Adam optimizer: gradient is null for a parameter");
        }

        size_t index = &param - &params[0]; // Get the index of the current parameter
        auto &m_t = m[index];
        auto &v_t = v[index];

        auto const &m_data = m_t->data();
        auto const &v_data = v_t->data();
        auto const &grad_data = grad->data();
        auto const &param_data = param->data();

        for (size_t i = 0; i < param->size(); ++i) {

            m_t->set_data(i, beta1 * m_data[i] + (1 - beta1) * grad_data[i]);
            v_t->set_data(i, beta2 * v_data[i] + (1 - beta2) * grad_data[i] * grad_data[i]);
        }

        for (size_t i = 0; i < param->size(); ++i) {
            float new_value = param_data[i] - learning_rate * m_t->data()[i] / (std::sqrt(v_t->data()[i]) + epsilon);
            param->set_data(i, new_value);
        }
    }
}