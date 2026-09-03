#include "nn/sequential.hpp"

Sequential::Sequential(const std::vector<std::shared_ptr<Module>> layers) {
    for (const auto &layer : layers) {

        add_module(layer);
    }
};

std::shared_ptr<Tensor> Sequential::forward(const std::shared_ptr<Tensor> &input) const {
    auto out = input;
    for (const auto &mod : sub_modules) {
        out = mod->forward(out);
    }

    return out;
}
