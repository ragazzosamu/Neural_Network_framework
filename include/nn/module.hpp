#pragma once
#include "core/tensor.hpp"
#include <memory>
#include <stdexcept>

/**
 * @brief Base class for all neural network building blocks (layers,
 *        activations, and composite networks).
 *
 * A Module owns its own learnable parameters (@ref params) and can contain
 * other Modules (@ref sub_modules), forming a tree. Operations that operate
 * on the whole tree — collecting parameters, listing sub-modules, or
 * switching between training and evaluation mode — recurse automatically
 * into every registered sub-module.
 *
 * Derived classes must implement forward(); everything else has a default
 * implementation based on @ref sub_modules and @ref params.
 */
class Module {

  public:
    /**
     * @brief Default virtual destructor.
     *
     * Ensures derived classes are destroyed correctly when deleted through
     * a base Module pointer.
     */
    virtual ~Module() = default;

    /**
     * @brief Computes this module's output for a given input tensor.
     *
     * @param input The input tensor to the module.
     * @return A new Tensor holding the result of the module's computation.
     *
     * @note Pure virtual: every concrete Module (e.g. Linear, Relu,
     *       Sequential) must provide its own forward() implementation, so
     *       the exact exceptions thrown depend on the derived class.
     */
    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const = 0;

    /**
     * @brief Registers a child module as a sub-module of this one.
     *
     * Sub-modules are included when collecting parameters() or modules(),
     * and their training mode is toggled together with this module's via
     * set_train() / set_eval().
     *
     * @param module The sub-module to register. Ownership is shared.
     *
     * @throws std::invalid_argument if @p module is null.
     */
    virtual void add_module(std::shared_ptr<Module> module) {
        if (!module) {
            throw std::invalid_argument("Module::add_module: cannot add a null module");
        }
        sub_modules.push_back(module);
    };

    /**
     * @brief Collects all learnable parameters owned by this module and,
     *        recursively, by every registered sub-module.
     *
     * @return A flat vector containing this module's own parameters
     *         (@ref params) followed by the parameters of each sub-module,
     *         in registration order.
     */
    virtual std::vector<std::shared_ptr<Tensor>> parameters() const {

        std::vector<std::shared_ptr<Tensor>> all_params = params;
        for (const auto &mod : sub_modules) {

            std::vector<std::shared_ptr<Tensor>> param_mod = mod->parameters();
            all_params.insert(all_params.end(), param_mod.begin(), param_mod.end());
        }

        return all_params;
    }

    /**
     * @brief Collects every sub-module in the tree rooted at this module.
     *
     * @return A flat vector of all direct and indirect sub-modules, in
     *         depth-first, registration order. This module itself is not
     *         included, only its descendants.
     */
    virtual std::vector<std::shared_ptr<Module>> modules() const {
        std::vector<std::shared_ptr<Module>> all_modules;
        for (const auto &mod : sub_modules) {
            all_modules.push_back(mod);
            auto children = mod->modules();
            all_modules.insert(all_modules.end(), children.begin(), children.end());
        }
        return all_modules;
    }

    /**
     * @brief Switches this module (and all its sub-modules) into training mode.
     *
     * If the module is already in training mode, this is a no-op. Otherwise,
     * every parameter returned by parameters() has its `requires_grad` flag
     * set to true via change_training_mode().
     */
    virtual void set_train() {

        // It changes the parameters only if it is in the other state
        if (!training_mode) {
            change_training_mode(true);
        }
    }

    /**
     * @brief Switches this module (and all its sub-modules) into evaluation mode.
     *
     * If the module is already in evaluation mode, this is a no-op.
     * Otherwise, every parameter returned by parameters() has its
     * `requires_grad` flag set to false via change_training_mode().
     */
    virtual void set_eval() {

        // It changes the parameters only if it is in the other state
        if (training_mode) {
            change_training_mode(false);
        }
    }

  protected:
    /// Whether this module is currently in training mode (true) or
    /// evaluation mode (false). Defaults to training mode.
    bool training_mode{true};

    /// Child modules registered via add_module(), used by parameters(),
    /// modules(), and change_training_mode() to recurse through the tree.
    std::vector<std::shared_ptr<Module>> sub_modules;

    /// Parameters owned directly by this module (not by its sub-modules).
    std::vector<std::shared_ptr<Tensor>> params;

    /**
     * @brief Sets `requires_grad` on every parameter in this module's tree.
     *
     * @param status The value to assign to each parameter's `requires_grad`
     *               flag: true when entering training mode, false when
     *               entering evaluation mode.
     *
     * @note Updates @ref training_mode is the caller's responsibility;
     *       this helper only propagates the flag to the parameters
     *       returned by parameters().
     */
    void change_training_mode(bool status) {
        auto all_params = this->parameters();
        for (auto &param : all_params) {
            param->set_requires_grad(status);
        }
        training_mode = status;
    }
};