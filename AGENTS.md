# Agent instructions

- This repository is currently documentation-only: the root contains `plan-spec.md` and the active implementation plan at `plans/tiny-int8-gemm.md`; there is no build system, source tree, test runner, CI configuration, or lockfile yet.
- Do not claim that planned commands, paths, targets, benchmarks, or dependencies exist until the implementation adds and verifies them. In particular, the `include/`, `src/`, `tests/`, `benchmarks/`, and `docs/` layout in the plan is future work, not the current repository layout.
- Before authoring or editing a plan, read `plan-spec.md`; it is the canonical format and terminology specification. Keep project-specific facts in this file or project docs, not in the generic spec.
- Treat `plans/tiny-int8-gemm.md` as the current intended execution order and scope, but verify its hardware, compiler, performance, and dependency assumptions when implementation begins.
- The planned implementation targets a single-threaded C11 `u8 × s8 → s32` x86-64 GEMM, with scalar/reference, blocked/packed, AVX2, and runtime-gated VNNI paths. Multi-threading, GPU, non-x86 targets, floating-point GEMM, and JIT code generation are explicitly out of scope unless the plan changes.
- Preserve the plan’s correctness and safety gates: bitwise agreement with the reference, sanitizer-clean memory behavior, warning-clean GCC and Clang builds, and no heap allocation in the hot GEMM path.
- Once executable tooling exists, update this file with exact verified setup, focused-test, sanitizer, benchmark, and formatting commands rather than copying aspirational commands from the plan.
- Current phase-1 verification uses GCC 9.4 and Clang 10 available in the environment: `./scripts/build.sh gcc`, then run tests with `(cd build && ctest --output-on-failure)`. Use `./scripts/build.sh clang` to switch compilers; the script recreates the single generated `build/` directory when necessary.
- `scripts/build.sh <compiler>` is the preferred build shortcut. It always configures an optimized Release build in `build/`; run tests with `(cd build && ctest --output-on-failure)`.
- Phase-2 loop-order profiling uses `./scripts/build.sh <compiler>` followed by `./build/profile_loops 512`. The profiler compares `i-j-k` and `i-k-j` using fenced x86 TSC reads and verifies equal outputs.


## Git branch selection

- At the start of a coding task, inspect the current branch and working tree. Follow an explicit branch name or instruction in the prompt; otherwise use the criteria below.
- Create a topic branch before editing for a standalone feature, bug fix, substantial refactor, or experiment when the current branch is the shared/default branch. Use a short task-based name such as `feature/<name>`, `fix/<name>`, `refactor/<name>`, or `experiment/<name>`.
- Stay on the current branch for small, localized changes, documentation-only work, follow-ups to work already on a suitable topic branch, or when the prompt clearly asks to continue the current branch.
- Never switch branches with unrelated uncommitted changes present, and never stash, discard, commit, or push changes unless asked. If a new branch seems warranted but the worktree is dirty, preserve it and ask before switching; otherwise continue on the current branch when safe.
- Do not create branches for informational questions. Do not commit or push as an automatic consequence of creating a branch.
