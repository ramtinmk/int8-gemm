# Tiny INT8 GEMM

Tiny INT8 GEMM is an educational, performance-oriented matrix multiplication library for x86-64 CPUs.
It computes row-major:

```text
C = A × B
A: M × K unsigned 8-bit integers
B: K × N signed 8-bit integers
C: M × N signed 32-bit integers
```

The project is intended to make the path from a clear scalar reference implementation to cache-aware and SIMD-optimized GEMM measurable and understandable. The long-term design follows the project plan in [`plans/tiny-int8-gemm.md`](plans/tiny-int8-gemm.md): loop reordering, cache blocking, packing, AVX2, VNNI dispatch, and comparative benchmarking.

## Current status

Implemented:

- C11 public API in [`include/gemm.h`](include/gemm.h)
- Scalar `i-j-k` reference implementation
- Scalar `i-k-j` loop-reordered implementation
- Deterministic correctness tests for non-square matrices
- `__rdtsc`-based loop-order profiler
- CMake targets for the library, tests, and benchmarks

The implementation is currently single-threaded and focused on x86-64. SIMD kernels, cache blocking, packing, and external-library comparisons are planned but not yet implemented.

## Build

The build helper accepts a compiler executable available on `PATH` and creates one optimized `build/` directory:

```bash
./scripts/build.sh gcc
```

To switch compilers, run the command again. The script recreates the generated `build/` directory when the compiler changes because CMake caches the compiler selection.

## Test

Run the CTest suite from the build directory:

```bash
(cd build && ctest --output-on-failure)
```

The test suite checks both scalar loop variants against an independently calculated reference result for `M=37`, `N=73`, and `K=129`.

## Profile loop order

Use an optimized build for performance measurements:

```bash
./scripts/build.sh gcc
./build/profile_loops 512
```

The profiler compares the `i-j-k` and `i-k-j` implementations on `512×512×512`, verifies that their outputs match, and reports fenced x86 TSC cycle counts. The optional dimension argument changes all three dimensions:

```bash
./build/profile_loops 256
```

Performance depends on compiler, CPU frequency, virtualization, and system load. Debug builds should not be used for these measurements.

## Other target

The standalone baseline benchmark can be run with:

```bash
./build/gemm_benchmark
```

## Scope

The planned implementation deliberately excludes multi-threading, GPU execution, non-x86 targets, floating-point GEMM, asymmetric zero-point handling in the compute loop, and JIT code generation. Correctness is defined as bitwise agreement with the scalar reference for results that fit in `int32_t`.

See [`AGENTS.md`](AGENTS.md) for repository-specific instructions and [`plan-spec.md`](plan-spec.md) for the plan authoring standard.
