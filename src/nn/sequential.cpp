#include "nn/sequential.hpp"

// Delegates entirely to Module::add_module(), which validates each layer
// (rejecting null ones) and appends it to sub_modules. Keeping construction
// this thin means Sequential doesn't need its own storage for the layer
// list — sub_modules, inherited from Module, is already the single source
// of truth used by forward(), parameters(), and modules().
Sequential::Sequential(const std::vector<std::shared_ptr<Module>> layers) {
    for (const auto &layer : layers) {

        add_module(layer);
    }
};

// Feeds the input through each layer in registration order, passing each
// layer's output as the next layer's input. If sub_modules is empty, the
// original input is returned unchanged rather than throwing — an empty
std::shared_ptr<Tensor> Sequential::forward(const std::shared_ptr<Tensor> &input) const {
    auto out = input;
    for (const auto &mod : sub_modules) {
        out = mod->forward(out);
    }

    return out;
}