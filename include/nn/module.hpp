#pragma once
#include "core/tensor.hpp"
#include <memory>
class Module {

  public:
    virtual ~Module() = default;

    virtual std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> &input) const = 0;

    virtual void add_module(std::shared_ptr<Module> module) { sub_modules.push_back(module); };

    virtual std::vector<std::shared_ptr<Tensor>> parameters() const {

        std::vector<std::shared_ptr<Tensor>> all_params = params;
        for (const auto &mod : sub_modules) {

            std::vector<std::shared_ptr<Tensor>> param_mod = mod->parameters();
            all_params.insert(all_params.end(), param_mod.begin(), param_mod.end());
        }

        return all_params;
    }

    virtual std::vector<std::shared_ptr<Module>> modules() const {
        std::vector<std::shared_ptr<Module>> all_modules;
        for (const auto &mod : sub_modules) {
            all_modules.push_back(mod);
            auto children = mod->modules();
            all_modules.insert(all_modules.end(), children.begin(), children.end());
        }
        return all_modules;
    }

    virtual void set_train() {

        // It changes the parameters only if it is in the other state
        if (!training_mode) {
            change_training_mode(true);
        }
    }
    virtual void set_eval() {

        // It changes the parameters only if it is in the other state
        if (training_mode) {
            change_training_mode(false);
        }
    }

  protected:
    bool training_mode{true};
    std::vector<std::shared_ptr<Module>> sub_modules;
    std::vector<std::shared_ptr<Tensor>> params;

    // helper
    void change_training_mode(bool status) {
        auto all_params = this->parameters();
        for (auto &param : all_params) {

            param->set_requires_grad(status);
        }
    }
};