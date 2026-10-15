#pragma once
#include "cuda/memory.hpp"

enum class Device { CPU, CUDA };

inline Device default_device() {
    if (cuda_is_available()) {
        return Device::CUDA;
    } else {
        return Device::CPU;
    }
}