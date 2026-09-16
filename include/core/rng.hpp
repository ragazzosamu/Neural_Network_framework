// core/rng.hpp
#pragma once
#include <random>

namespace rng {
void set_seed(unsigned int s);
std::mt19937 &engine();
unsigned int seed();
} // namespace rng