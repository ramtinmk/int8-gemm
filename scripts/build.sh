#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <compiler>\n' "$0" >&2
    printf 'Example: %s gcc\n' "$0" >&2
    exit 2
fi

compiler=$1
if ! command -v "$compiler" >/dev/null 2>&1; then
    printf 'Compiler not found: %s\n' "$compiler" >&2
    exit 1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
compiler_path=$(command -v "$compiler")
build_dir="$repo_root/build"

# CMake cannot change compilers in an existing build tree. Recreate only the
# generated build directory when the requested compiler differs.
cache_file="$build_dir/CMakeCache.txt"
if [[ -f "$cache_file" ]]; then
    configured_compiler=$(sed -n 's/^CMAKE_C_COMPILER:[^=]*=//p' "$cache_file")
    configured_build_type=$(sed -n 's/^CMAKE_BUILD_TYPE:[^=]*=//p' "$cache_file")
    if [[ "$configured_compiler" != "$compiler_path" || "$configured_build_type" != "Release" ]]; then
        rm -rf "$build_dir"
    fi
fi

cmake -S "$repo_root" \
      -B "$build_dir" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="$compiler_path"
cmake --build "$build_dir" --parallel
