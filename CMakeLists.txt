cmake_minimum_required(VERSION 3.15)
project(Neural_Networks_framework VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
include(FetchContent)


FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark) #benchmark::benchmark

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2) #Catch2::Catch2


add_executable(EseguibileTest tests/test_iniziale.cpp)
add_executable(Catch2Test tests/test_Catch2.cpp)
target_link_libraries(Catch2Test PRIVATE Catch2::Catch2WithMain)

