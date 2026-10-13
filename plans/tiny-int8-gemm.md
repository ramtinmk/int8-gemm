---
plan_id: tiny-int8-gemm
title: High-Performance INT8 GEMM Engine on x86-64 AVX2/VNNI
category: program
status: active
owner: Ramtin Mahmoudi Kashani
created: 2026-10-13
updated: 2026-10-13
related: []
---


# High-Performance INT8 GEMM Engine on x86-64 AVX2/VNNI

## Purpose

Design, implement, and benchmark an educational yet production-comparable INT8 general matrix multiplication (GEMM) library for modern x86-64 architectures. The project implements the analytical GotoBLAS/BLIS five-loop cache hierarchy, explicit memory packing, and SIMD microkernels (AVX2 and VNNI) on an 11th Gen Intel Core i5 under WSL 2, producing an end-to-end performance study directly evaluated against Intel oneDNN and Google `ruy`.

## Scope

* **In:**
* C11 library implementing row-major unsigned/signed integer matrix multiplication ($u8 \times s8 \to s32$).
* Reference scalar implementations (naive row-major triple loop and loop-reordered $i\text{-}k\text{-}j$).
* GotoBLAS cache blocking and memory packing routines for matrices $A$ and $B$.
* Handcrafted vector microkernels targeting x86-64 AVX2 (`_mm256_maddubs_epi16`, `_mm256_madd_epi16`) and AVX-512/AVX-VNNI (`_mm256_dpbusd_epi32`).
* Google Benchmark harness measuring cycle latency, throughput (GOPS), and memory bandwidth.
* Verification suite with bitwise equality checks against a multi-precision software reference.


* **Out:**
* Multi-threaded parallelization via OpenMP or `pthread` pools (single-core micro-architecture focus).
* General floating-point GEMM (FP32/FP64) or asymmetric zero-point quantization shifts in the compute loop.
* JIT runtime code generation (all kernels are statically compiled C11 with intrinsics).
* GPU/CUDA execution.


* **Work location:** Standalone repository `tiny-int8-gemm/` containing `include/`, `src/`, `tests/`, `benchmarks/`, and `docs/`.

## Baseline

* **Observed starting state:** Unoptimized scalar naive triple-loop ($i, j, k$) GEMM compiled with `-O2` on a single physical core of an Intel Core i5-1135G7 (WSL 2 Ubuntu Linux kernel).
* **Evidence:** Running $M = N = K = 512$ with random inputs yields an execution time of $318.5\text{ ms}$, corresponding to $0.84\text{ GOPS}$ at an effective single-core clock of $3.8\text{ GHz}$. This represents approximately $0.31\%$ of the CPU's theoretical arithmetic integer peak.

## Goals and Non-goals

* **Goals:**
* **G1:** Zero numerical discrepancies (exact bitwise parity across all output elements) between vector kernels and the software reference implementation across matrix dimensions $M, N, K \in [16, 2048]$.
* **G2:** Achieve $\ge 80.0\text{ GOPS}$ single-thread throughput for $M = N = K = 1024$ using the AVX2 microkernel on Intel Core i5-1135G7.
* **G3:** Achieve $\ge 120.0\text{ GOPS}$ single-thread throughput for $M = N = K = 1024$ using the VNNI microkernel on supported 11th Gen hardware.
* **G4:** Achieve $\ge 60\%$ of Intel oneDNN (`dnnl_gemm_u8s8s32`) single-threaded throughput across square matrix sizes $N \in \{256, 512, 1024, 2048\}$.
* **G5:** Zero heap allocations inside the hot GEMM execution path (all packing buffers statically pre-allocated or arena-managed).


* **Non-goals:**
* **NG1:** Support for non-x86 architectures (e.g., ARM NEON, RISC-V Vector).
* **NG2:** Multi-core thread scaling or dynamic work-stealing schedulers.
* **NG3:** End-to-end neural network model execution or ONNX graph parsing.



## Measurement Contract

* **Metric:** Effective Throughput in GOPS, calculated strictly as:

$$\text{GOPS} = \frac{2 \times M \times N \times K}{\text{time in seconds} \times 10^9}$$


* **Hardware Environment:** Single physical core, Intel Core i5 11th Gen (Tiger Lake), 8 GB RAM, WSL 2 (Ubuntu 22.04 LTS), CPU frequency governor set to `performance`.
* **Compiler Configuration:** GCC 12 or Clang 15 with compilation flags:
`-O3 -mavx2 -mfma -mavxvnni -march=tigerlake -Wall -Wextra -Werror -pedantic`
* **Execution Windows:**
* *Warmup Window:* 10 unmeasured iterations to ensure L1/L2 cache residency of instructions and frequency stabilization.
* *Measurement Window:* Minimum 50 iterations or $2.0$ seconds continuous steady-state execution time per matrix dimension.
* *Drain Window:* None (synchronous in-memory computation).


* **Reported Fields:** Matrix dimensions $(M, N, K)$, `real_time_ms`, `cpu_time_ms`, `gops`, `cycles_per_op` (sampled via `__rdtsc`), and percentage of hardware theoretical peak.

## Progress Snapshot

* [x] Baseline established and hardware target verified.
* [~] `tiny-int8-gemm/phase-1-scaffolding`: Project scaffolding and reference implementation (validated with installed GCC 9.4 and Clang 10; GCC 12 and Clang 15 are not installed).
* [x] `tiny-int8-gemm/phase-2-loop-reordering`: Memory access loop reordering and memory stride analysis.
* [ ] `tiny-int8-gemm/phase-3-cache-blocking`: Five-loop BLIS cache partitioning without packing.
* [ ] `tiny-int8-gemm/phase-4-memory-packing`: Continuous buffer memory packing routines (`PackA`, `PackB`).
* [ ] `tiny-int8-gemm/phase-5-avx2-microkernel`: AVX2 integer dot-product microkernel.
* [ ] `tiny-int8-gemm/phase-6-vnni-specialization`: Intel VNNI instruction path specialization.
* [ ] `tiny-int8-gemm/phase-7-onednn-benchmark`: Comparative benchmark suite against Intel oneDNN and Google `ruy`.

---

## Phases

### tiny-int8-gemm/phase-1-scaffolding: Project Scaffolding and Reference Implementation

* **Objective:** Establish the build system, API surface, test framework, and high-precision reference implementation.
* **Complexity:** 2 — Standard directory structure, CMake configuration, and reference logic without low-level optimizations.
* **Risk:** Low — Well-understood patterns with zero hardware-specific dependencies.

**Work**

* [x] Define canonical C11 function signature `gemm_u8s8s32` in `include/gemm.h`.
* [x] Create CMake build system supporting targets for library, tests, and benchmarks with `-Wall -Wextra -Werror`.
* [x] Implement scalar reference GEMM in `src/ref/gemm_reference.c` utilizing 64-bit integer accumulation to prevent intermediate overflow.
* [x] Implement test runner using synthetic matrices populated with deterministic pseudorandom data.

**Exit criteria**

* [ ] Build completes without warnings under both GCC 12 and Clang 15 via `cmake --build build`.
* [x] Test suite executes reference implementation on non-square dimensions $(M=37, N=73, K=129)$ and verifies result against mathematically computed values; evidence: `ctest --output-on-failure` exit code `0` in both configured build directories.

The compiler-version-specific build criterion remains open because GCC 12 and Clang 15 are unavailable in the current environment.

---

### tiny-int8-gemm/phase-2-loop-reordering: Memory Access Loop Reordering

* **Objective:** Quantify and eliminate cache misses from non-contiguous memory strides by reordering loops from $(i, j, k)$ to $(i, k, j)$.
* **Complexity:** 2 — Minimal code modifications across single module; primary effort is profiling.
* **Risk:** Low — Pure algorithmic transformation without assembly or intrinsics.

**Work**

* [x] Implement naive triple-loop $(i, j, k)$ in `src/scalar/gemm_ijk.c`.
* [x] Implement reordered triple-loop $(i, k, j)$ in `src/scalar/gemm_ikj.c` enabling unit-stride access across rows of $B$ and $C$.
* [x] Add standalone profiling binary utilizing `__rdtsc` cycle timing.

**Exit criteria**

* [x] Output of `gemm_ikj` matches `gemm_reference` with $0$ bitwise discrepancies across all elements for $M = N = K = 512$; evidence: `gemm_tests` covers the loop variants on non-square dimensions and `profile_loops 512` verifies equal outputs.
* [x] Execution of `gemm_ikj` demonstrates at least a $4.0\times$ speedup over `gemm_ijk` on $M = N = K = 512$ in the GCC Release build; evidence: `profile_loops` reported `5.21x` with GCC and `4.15x` with Clang 10 using the single `build/` directory.

---

### tiny-int8-gemm/phase-3-cache-blocking: Five-Loop BLIS Cache Partitioning

* **Objective:** Implement the GotoBLAS/BLIS analytical cache hierarchy partitioning inputs to fit into L1, L2, and L3 cache boundaries without SIMD.
* **Complexity:** 3 — Complex index arithmetic across five nested loops; requires careful calculation of architecture cache sizes.
* **Risk:** Medium — Off-by-one errors and edge-case boundary conditions when matrix sizes are not multiples of block sizes.

**Work**

* [ ] Define analytical cache parameters for Intel Tiger Lake: $M_C = 256$, $K_C = 256$, $N_C = 1024$ in `include/gemm_config.h`.
* [ ] Implement the five-loop control flow in `src/blocked/gemm_blocked.c` traversing $N_C$ (Loop 5), $K_C$ (Loop 4), $M_C$ (Loop 3), $N_R$ (Loop 2), and $M_R$ (Loop 1).
* [ ] Handle arbitrary non-multiple dimensions via fringe/edge cleanup loops.

**Exit criteria**

* [ ] Bitwise equivalence verified against `gemm_reference` on prime matrix dimensions $M = 317, N = 521, K = 257$; evidence: test suite log.
* [ ] Throughput of `gemm_blocked` exceeds `gemm_ikj` by at least $1.5\times$ on large matrices where data spills past L3 cache ($M = N = K = 2048$); evidence: benchmark output table.

---

### tiny-int8-gemm/phase-4-memory-packing: Continuous Buffer Memory Packing

* **Objective:** Implement cache-friendly buffer packing routines for blocks of matrices $A$ and $B$ to ensure contiguous linear memory access in inner kernels.
* **Complexity:** 3 — Memory layout transformation involving panel interleaving and cache-line alignment.
* **Risk:** Medium — Memory access violations or performance degradation if packing overhead exceeds memory-access savings.

**Work**

* [ ] Implement `pack_b` routine in `src/packing/pack_b.c` storing panels of $B$ of size $K_C \times N_R$ contiguously in row-major or micro-panel layout.
* [ ] Implement `pack_a` routine in `src/packing/pack_a.c` storing panels of $A$ of size $M_C \times K_C$ in column-panel layout.
* [ ] Integrate a static aligned workspace allocator (using `posix_memalign` with 64-byte alignment) to avoid dynamic allocations during execution.

**Exit criteria**

* [ ] Memory verification check confirms all packed buffers remain aligned to 64-byte boundaries; evidence: programmatic assertion `((uintptr_t)ptr % 64) == 0`.
* [ ] AddressSanitizer run confirms zero buffer over-reads or out-of-bounds writes during packing of unaligned matrix sizes $(M=127, N=255, K=511)$; evidence: `ASAN_OPTIONS=strict_string_checks=1` clean exit.

---

### tiny-int8-gemm/phase-5-avx2-microkernel: AVX2 Integer Dot-Product Microkernel

* **Objective:** Construct an $M_R \times N_R$ register-blocked compute microkernel in C utilizing x86 AVX2 vector intrinsics.
* **Complexity:** 3 — Low-level register budget management (16 `ymm` registers), vector unrolling, and two-stage INT8 multiply-accumulate logic.
* **Risk:** High — Compiler register spilling if register allocation is unoptimal; arithmetic overflow if sign extensions are incorrect.

**Work**

* [ ] Design register layout for tile $M_R = 4, N_R = 16$ utilizing 8 accumulator registers (`ymm0`–`ymm7`), 2 registers for broadcasting elements of $A$, and 2 registers for streaming rows of packed $B$.
* [ ] Implement the two-step accumulation using `_mm256_maddubs_epi16` ($u8 \times s8 \to s16$) followed by `_mm256_madd_epi16` with a ones-vector to accumulate into signed 32-bit integer lanes.
* [ ] Implement microkernel function `gemm_kernel_avx2_4x16` in `src/kernels/kernel_avx2.c`.
* [ ] Wire the microkernel into the 5-loop blocked execution engine.

**Exit criteria**

* [ ] Output of full AVX2 GEMM exhibits $0$ bitwise discrepancies compared to `gemm_reference` across 1,000 randomized matrices with entries spanning full ranges $u8 \in [0, 255]$ and $s8 \in [-128, 127]$; evidence: `test_correctness` passing report.
* [ ] Single-thread throughput reaches $\ge 80.0\text{ GOPS}$ on $M = N = K = 1024$ on Intel Core i5-1135G7; evidence: Google Benchmark report column `gops`.

---

### tiny-int8-gemm/phase-6-vnni-specialization: Intel VNNI Instruction Path Specialization

* **Objective:** Implement a dedicated microkernel path utilizing the single-cycle `vpdpbusd` instruction via AVX-512 VNNI or AVX-VNNI.
* **Complexity:** 2 — Replaces the two-instruction AVX2 multiply-accumulate sequence with `_mm256_dpbusd_epi32`; builds directly on Phase 5 framework.
* **Risk:** Low — Algorithmic structure is identical to Phase 5; verified via hardware feature checks.

**Work**

* [ ] Implement runtime CPUID feature detection for `AVX_VNNI` and `AVX512_VNNI` in `src/cpu_features.c`.
* [ ] Implement `gemm_kernel_vnni_4x16` in `src/kernels/kernel_vnni.c` utilizing `_mm256_dpbusd_epi32`.
* [ ] Provide dynamic dispatch mechanism selecting VNNI if available, otherwise falling back to Phase 5 AVX2 kernel.

**Exit criteria**

* [ ] CPUID detection properly identifies Tiger Lake hardware capabilities; evidence: logged feature string on startup matches `/proc/cpuinfo` flags.
* [ ] VNNI kernel achieves at least a $1.4\times$ throughput speedup over the Phase 5 AVX2 kernel on $M = N = K = 1024$; evidence: benchmark comparison table.

---

### tiny-int8-gemm/phase-7-onednn-benchmark: Comparative Benchmark Suite

* **Objective:** Build a unified benchmark driver comparing this implementation against Intel oneDNN and Google `ruy`, and generate visualization artifacts.
* **Complexity:** 3 — Integrating external dependencies (oneDNN C API, Google Benchmark) and structuring comparative telemetry.
* **Risk:** Medium — Divergent quantization post-op conventions between libraries requiring normalization of compute time.

**Work**

* [ ] Integrate Intel oneDNN (`dnnl_gemm_u8s8s32`) into benchmark harness in `benchmarks/bench_comparison.cpp`.
* [ ] Structure test cases covering square matrices $N \in \{64, 128, 256, 512, 1024, 2048\}$ and Transformer decode shapes $(M=1, K=4096, N=4096)$ and prefill shapes $(M=128, K=4096, N=4096)$.
* [ ] Write Python visualization script `benchmarks/plot_roofline.py` generating roofline analysis and GOPS comparison charts.
* [ ] Embed benchmark results, hardware specs, and analysis into project `README.md`.

**Exit criteria**

* [ ] Benchmark suite successfully executes without crashes across all declared shapes for both `tiny-int8-gemm` and Intel oneDNN; evidence: Google Benchmark JSON output file.
* [ ] Achieved throughput meets or exceeds $60\%$ of Intel oneDNN single-thread performance on $M = N = K = 1024$; evidence: benchmark JSON comparison summary.
* [ ] Execution of `python benchmarks/plot_roofline.py` produces publication-grade SVG/PNG charts visualizing throughput versus theoretical peak; evidence: generated files exist in `docs/assets/`.

---

## Gates

* **G-1 (Correctness):** Zero bitwise errors tolerated against reference output. Any regression immediately blocks commits.
* **G-2 (Safety):** Zero memory leaks or invalid accesses under AddressSanitizer and UndefinedBehaviorSanitizer across all test matrix dimensions.
* **G-3 (Portability & Tooling):** Clean compilation under `-Wall -Wextra -Werror -pedantic` with both Clang and GCC. No Microsoft/GNU non-standard C extensions permitted in library headers.
* **G-4 (Zero Heap Thrashing):** Zero calls to `malloc`, `calloc`, or `free` inside the compute loop; verified via heap profiling or custom memory hooks.

## Risks and Rollback

| Risk | Likelihood | Impact | Mitigation / Rollback Plan |
| --- | --- | --- | --- |
| **Compiler spills registers in AVX2 microkernel**, dropping throughput by $>50\%$. | High | High | Inspect disassembly (`objdump -d`). If spilling occurs, reduce tile height $M_R$ from 4 to 2, or implement the inner microkernel via direct GNU inline assembly (`__asm__ volatile`) to guarantee register pinning. |
| **WSL 2 dynamic virtualization overhead / clock jitter** degrades timing reproducibility. | Medium | Medium | Pin benchmark threads to core 0 via `pthread_setaffinity_np`; report hardware CPU cycle counts (`__rdtsc`) as the primary invariant metric alongside wall-clock time. |
| **Tiger Lake CPU SKU lacks specific VNNI variant** (`AVX512_VNNI` vs `AVX_VNNI`). | Medium | Low | Feature gate the compilation: compile VNNI kernel only when compiler flag succeeds, and runtime CPUID safely routes to the AVX2 kernel without crashing. |
| **Negative performance from buffer packing overhead** on smaller matrix dimensions ($N < 128$). | High | Low | Introduce an analytical threshold in the driver: bypass packing and use a direct in-place microkernel when matrix dimensions fit entirely within L1 cache ($M \times K + K \times N < 32\text{ KB}$). |

## Execution Order

The execution of phases is strictly sequential due to structural dependencies:

1. `tiny-int8-gemm/phase-1-scaffolding`
2. `tiny-int8-gemm/phase-2-loop-reordering`
3. `tiny-int8-gemm/phase-3-cache-blocking`
4. `tiny-int8-gemm/phase-4-memory-packing`
5. `tiny-int8-gemm/phase-5-avx2-microkernel`
6. `tiny-int8-gemm/phase-6-vnni-specialization`
7. `tiny-int8-gemm/phase-7-onednn-benchmark`
