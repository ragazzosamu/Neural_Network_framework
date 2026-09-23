cmake_minimum_required(VERSION 3.15)
project(Neural_Networks_framework VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

# Keep every FetchContent dependency inside the selected build directory.
# This makes `build` self-contained and avoids creating/using sibling build
# directories for downloaded sources and generated dependency files.
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps" CACHE PATH
    "Directory used for all FetchContent dependencies")


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
            src/core/tensor.cpp
            src/core/rng.cpp)
target_include_directories(tensor_lib PUBLIC include)

add_library(operations_lib
            src/ops/matadd.cpp
            src/ops/matmul.cpp
            src/ops/relu.cpp
            src/ops/softmax.cpp
            src/ops/crossentropy.cpp
            src/ops/mean.cpp
            src/ops/mse.cpp
            src/ops/im2col.cpp
            src/ops/col2im.cpp
            src/ops/flatten_convolution_output.cpp
            src/ops/weight2col.cpp)
target_include_directories(operations_lib PUBLIC include)
target_link_libraries(operations_lib PUBLIC tensor_lib)

option(NN_USE_OPENBLAS "Use OpenBLAS for 2D operations" OFF)
option(NN_FETCH_OPENBLAS "Download OpenBLAS with FetchContent" ON)

if(NN_USE_OPENBLAS)
    if(NN_FETCH_OPENBLAS)
        set(BUILD_TESTING OFF CACHE BOOL "Disable OpenBLAS tests" FORCE)
        set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable OpenBLAS benchmarks" FORCE)
        set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "Build only BLAS and CBLAS" FORCE)
        set(ONLY_CBLAS ON CACHE BOOL "Build only the CBLAS interface" FORCE)

        FetchContent_Declare(
            openblas
            GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
            GIT_TAG v0.3.29
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(openblas)

        if(TARGET openblas)
            set(OPENBLAS_TARGET openblas)
        elseif(TARGET OpenBLAS::OpenBLAS)
            set(OPENBLAS_TARGET OpenBLAS::OpenBLAS)
        else()
            message(FATAL_ERROR "FetchContent downloaded OpenBLAS, but no CMake target was created.")
        endif()

        target_include_directories(operations_lib PRIVATE
            ${openblas_SOURCE_DIR}
            ${openblas_SOURCE_DIR}/include
            ${openblas_SOURCE_DIR}/lapack-netlib/CBLAS/include
            ${openblas_BINARY_DIR}
            ${openblas_BINARY_DIR}/include
            ${openblas_BINARY_DIR}/include/openblas
        )
        target_link_libraries(operations_lib PRIVATE ${OPENBLAS_TARGET})
    else()
        set(OPENBLAS_ROOT "" CACHE PATH "OpenBLAS installation directory")

        find_path(OPENBLAS_INCLUDE_DIR
            NAMES cblas.h
            HINTS ${OPENBLAS_ROOT}
            PATH_SUFFIXES include
        )
        find_library(OPENBLAS_LIBRARY
            NAMES openblas libopenblas
            HINTS ${OPENBLAS_ROOT}
            PATH_SUFFIXES lib
        )

        if(NOT OPENBLAS_INCLUDE_DIR OR NOT OPENBLAS_LIBRARY)
            message(FATAL_ERROR
                "OpenBLAS was not found. Set OPENBLAS_ROOT or enable NN_FETCH_OPENBLAS."
            )
        endif()

        target_include_directories(operations_lib PRIVATE ${OPENBLAS_INCLUDE_DIR})
        target_link_libraries(operations_lib PRIVATE ${OPENBLAS_LIBRARY})
    endif()

    target_compile_definitions(operations_lib PRIVATE NN_USE_OPENBLAS)
endif()

add_library(loss_lib
            src/loss.cpp)
            
target_include_directories(loss_lib PUBLIC include)
target_link_libraries(loss_lib PUBLIC operations_lib)
target_link_libraries(loss_lib PUBLIC tensor_lib)

add_library(nn_lib
            src/nn/conv2D.cpp
            src/nn/linear.cpp
            src/nn/sequential.cpp
            src/nn/relu.cpp)

target_include_directories(nn_lib PUBLIC include)
target_link_libraries(nn_lib PUBLIC operations_lib)
target_link_libraries(nn_lib PUBLIC tensor_lib)

add_library(optimizer_lib
            src/optimizer/sgd.cpp
            src/optimizer/adam.cpp)
target_include_directories(optimizer_lib PUBLIC include)
target_link_libraries(optimizer_lib PUBLIC tensor_lib)

add_executable(TensorTest tests/test_tensor.cpp)
target_link_libraries(TensorTest PRIVATE tensor_lib)
target_link_libraries(TensorTest PRIVATE Catch2::Catch2WithMain)

add_executable(OperationTest tests/test_operations.cpp)
target_link_libraries(OperationTest PRIVATE operations_lib tensor_lib)
target_link_libraries(OperationTest PRIVATE Catch2::Catch2WithMain)


add_executable(LossTest tests/test_loss.cpp)
target_link_libraries(LossTest PRIVATE loss_lib)
target_link_libraries(LossTest PRIVATE Catch2::Catch2WithMain)

add_executable(ModuleTest tests/test_modules.cpp)
target_link_libraries(ModuleTest PRIVATE nn_lib)
target_link_libraries(ModuleTest PRIVATE Catch2::Catch2WithMain)

add_executable(OptimizerTest tests/test_optimizer.cpp)
target_link_libraries(OptimizerTest PRIVATE optimizer_lib nn_lib operations_lib tensor_lib)
target_link_libraries(OptimizerTest PRIVATE Catch2::Catch2WithMain)

add_executable(xor examples/cpp/xor.cpp)
target_link_libraries(xor PRIVATE optimizer_lib nn_lib operations_lib tensor_lib loss_lib)
target_link_libraries(xor PRIVATE Catch2::Catch2WithMain)
target_compile_definitions(xor PRIVATE NEURAL_NETWORK_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(mnist examples/cpp/mnist/mnist.cpp)
target_link_libraries(mnist PRIVATE optimizer_lib nn_lib loss_lib operations_lib tensor_lib)
target_compile_definitions(mnist PRIVATE NEURAL_NETWORK_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

add_executable(benchmarc_mnist examples/cpp/mnist/benchmarc_mnist.cpp)
target_link_libraries(benchmarc_mnist PRIVATE optimizer_lib nn_lib loss_lib operations_lib tensor_lib Catch2::Catch2WithMain)
target_compile_definitions(benchmarc_mnist PRIVATE NEURAL_NETWORK_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")



