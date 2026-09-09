// core/rng.cpp
#include "core/rng.hpp"
#include <iostream>

namespace {
unsigned int g_seed = 0;
std::mt19937 g_engine;
bool g_initialized = false;
} // namespace

namespace rng {
void set_seed(unsigned int s) {
    g_seed = s;
    g_engine.seed(s);
    g_initialized = true;
    std::cout << "[RNG] seed impostato a: " << g_seed << std::endl;
}

std::mt19937 &engine() {
    if (!g_initialized) {
        std::random_device rd;
        set_seed(rd());
    }
    return g_engine;
}

unsigned int seed() { return g_seed; }
} // namespace rng