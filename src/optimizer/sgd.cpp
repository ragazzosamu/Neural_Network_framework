#include "optimizer/sgd.hpp"

void SGD::step() {

    for (auto const &param : params) {
        auto &grad = param->get_grad();
        if (grad == nullptr) {
            throw std::runtime_error("SGD optimizer: gradient is null for a parameter");
        }

        float *param_data = param->data();
        const float *grad_data = grad->data();
        const size_t size = param->size();

        for (size_t i = 0; i < size; ++i) {
            param_data[i] -= learning_rate * grad_data[i];
        }
    }
}