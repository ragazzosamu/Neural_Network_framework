#include "optimizer/sgd.hpp"

void SGD::step() {

    for (auto const &param : params) {
        auto &grad = param->get_grad();
        if (grad == nullptr) {
            throw std::runtime_error("SGD optimizer: gradient is null for a parameter");
        }

        auto const &param_data = param->data();
        auto const &grad_data = grad->data();

        for (size_t i = 0; i < param->size(); ++i) {
            float new_value = param_data[i] - learning_rate * grad_data[i];
            param->set_data(i, new_value);
        }
    }
}