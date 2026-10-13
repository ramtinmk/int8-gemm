#include "gemm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

static uint32_t next_random(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static uint64_t read_cycles_start(void)
{
    _mm_lfence();
    return __rdtsc();
}

static uint64_t read_cycles_end(void)
{
    unsigned int auxiliary;
    const uint64_t cycles = __rdtscp(&auxiliary);

    _mm_lfence();
    return cycles;
}

static int outputs_match(const int32_t *left, const int32_t *right, size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index) {
        if (left[index] != right[index]) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    const size_t dimension = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 512;
    const size_t m = dimension;
    const size_t n = dimension;
    const size_t k = dimension;
    const size_t a_count = m * k;
    const size_t b_count = k * n;
    const size_t c_count = m * n;
    uint8_t *a = malloc(a_count * sizeof(*a));
    int8_t *b = malloc(b_count * sizeof(*b));
    int32_t *ijk = calloc(c_count, sizeof(*ijk));
    int32_t *ikj = calloc(c_count, sizeof(*ikj));
    uint32_t state = UINT32_C(0x12345678);
    size_t index;
    uint64_t start;
    uint64_t ijk_cycles;
    uint64_t ikj_cycles;

    if (a == NULL || b == NULL || ijk == NULL || ikj == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(ijk);
        free(ikj);
        return EXIT_FAILURE;
    }

    for (index = 0; index < a_count; ++index) {
        a[index] = (uint8_t)(next_random(&state) & UINT32_C(0xff));
    }
    for (index = 0; index < b_count; ++index) {
        b[index] = (int8_t)(next_random(&state) & UINT32_C(0xff));
    }

    start = read_cycles_start();
    gemm_u8s8s32_ijk(a, b, ijk, m, n, k);
    ijk_cycles = read_cycles_end() - start;

    start = read_cycles_start();
    gemm_u8s8s32_ikj(a, b, ikj, m, n, k);
    ikj_cycles = read_cycles_end() - start;

    if (!outputs_match(ijk, ikj, c_count)) {
        fprintf(stderr, "loop variants produced different outputs\n");
        free(a);
        free(b);
        free(ijk);
        free(ikj);
        return EXIT_FAILURE;
    }

    printf("M=%zu N=%zu K=%zu ijk_cycles=%llu ikj_cycles=%llu speedup=%.2fx\n",
           m,
           n,
           k,
           (unsigned long long)ijk_cycles,
           (unsigned long long)ikj_cycles,
           (double)ijk_cycles / (double)ikj_cycles);

    free(a);
    free(b);
    free(ijk);
    free(ikj);
    return EXIT_SUCCESS;
}
