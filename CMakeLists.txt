cmake_minimum_required(VERSION 3.15)
project(Neural_Networks_framework VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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

add_library(tensor_lib
            src/tensor.cpp)
target_include_directories(tensor_lib PUBLIC include)

add_library(operations_lib
            src/matsum.cpp
            src/matmul.cpp
            src/relu.cpp
            src/softmax.cpp
            src/crossentropy.cpp
            src/average.cpp)
target_include_directories(operations_lib PUBLIC include)

add_library(loss_lib
            src/loss.cpp)

target_include_directories(loss_lib PUBLIC include)
target_link_libraries(loss_lib PUBLIC operations_lib)
target_link_libraries(loss_lib PUBLIC tensor_lib)

add_executable(TensorTest tests/test_tensor.cpp)
target_link_libraries(TensorTest PRIVATE tensor_lib)
target_link_libraries(TensorTest PRIVATE Catch2::Catch2WithMain)

add_executable(OperationTest tests/test_operations.cpp)
target_link_libraries(operations_lib PUBLIC tensor_lib)
target_link_libraries(OperationTest PRIVATE operations_lib tensor_lib)
target_link_libraries(OperationTest PRIVATE Catch2::Catch2WithMain)


add_executable(LossTest tests/test_loss.cpp)
target_link_libraries(LossTest PRIVATE loss_lib)
target_link_libraries(LossTest PRIVATE Catch2::Catch2WithMain)


