cmake_minimum_required(VERSION 3.16)

project(tiny_int8_gemm VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(TINY_INT8_GEMM_BUILD_BENCHMARKS "Build the standalone benchmark" ON)

set(TINY_INT8_GEMM_WARNINGS
    -Wall
    -Wextra
    -Werror
    -pedantic
)

add_library(tiny_int8_gemm
    src/ref/gemm_reference.c
    src/scalar/gemm_ijk.c
    src/scalar/gemm_ikj.c
)

target_include_directories(tiny_int8_gemm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(tiny_int8_gemm PRIVATE ${TINY_INT8_GEMM_WARNINGS})

enable_testing()
add_executable(gemm_tests tests/gemm_tests.c)
target_link_libraries(gemm_tests PRIVATE tiny_int8_gemm)
target_compile_options(gemm_tests PRIVATE ${TINY_INT8_GEMM_WARNINGS})
add_test(NAME gemm_tests COMMAND gemm_tests)

if(TINY_INT8_GEMM_BUILD_BENCHMARKS)
    add_executable(gemm_benchmark benchmarks/gemm_benchmark.c)
    target_link_libraries(gemm_benchmark PRIVATE tiny_int8_gemm)
    target_compile_options(gemm_benchmark PRIVATE ${TINY_INT8_GEMM_WARNINGS})

    add_executable(profile_loops benchmarks/profile_loops.c)
    target_link_libraries(profile_loops PRIVATE tiny_int8_gemm)
    target_compile_options(profile_loops PRIVATE ${TINY_INT8_GEMM_WARNINGS})
endif()
