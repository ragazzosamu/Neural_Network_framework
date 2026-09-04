#pragma once
#include "nn/module.hpp"

/**
 * @brief Container module that chains a list of layers, feeding each one's
 *        output into the next.
 *
 * All layers passed to the constructor are registered as sub-modules via
 * Module::add_module(), so they are automatically included in
 * parameters(), modules(), and set_train()/set_eval() without Sequential
 * needing to override any of them.
 */
class Sequential : public Module {

  public:
    /**
     * @brief Constructs a Sequential module from an ordered list of layers.
     *
     * @param layers The layers to chain together, in the order they should
     *        be applied during forward(). Each layer is registered via
     *        add_module().
     *
     * @throws std::invalid_argument if any element of @p layers is null
     *         (propagated from Module::add_module()).
     */
    explicit Sequential(const std::vector<std::shared_ptr<Module>> layers);

    /**
     * @brief Runs the input through every registered layer, in order.
     *
     * @param input The input tensor to the first layer.
     * @return The output of the last layer, or the unmodified @p input if
     *         no layers were registered.
     *
     * @throws Whatever exception the first failing layer's forward() throws;
     *         Sequential does not catch or wrap it.
     */
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const override;
};