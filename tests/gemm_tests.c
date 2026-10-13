#include "gemm.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t next_random(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static int check_non_square_case(void)
{
    const size_t m = 37;
    const size_t n = 73;
    const size_t k = 129;
    const size_t a_count = m * k;
    const size_t b_count = k * n;
    const size_t c_count = m * n;
    uint8_t *a = malloc(a_count * sizeof(*a));
    int8_t *b = malloc(b_count * sizeof(*b));
    int32_t *c = calloc(c_count, sizeof(*c));
    uint32_t state = UINT32_C(0x12345678);
    size_t index;

    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    for (index = 0; index < a_count; ++index) {
        a[index] = (uint8_t)(next_random(&state) & UINT32_C(0xff));
    }
    for (index = 0; index < b_count; ++index) {
        b[index] = (int8_t)(next_random(&state) & UINT32_C(0xff));
    }

    gemm_u8s8s32(a, b, c, m, n, k);

    for (index = 0; index < c_count; ++index) {
        const size_t row = index / n;
        const size_t column = index % n;
        int64_t expected = 0;
        size_t inner;

        for (inner = 0; inner < k; ++inner) {
            expected += (int64_t)a[row * k + inner] *
                        (int64_t)b[inner * n + column];
        }
        if (c[index] != (int32_t)expected) {
            fprintf(stderr,
                    "mismatch at (%zu, %zu): got %" PRId32 ", expected %" PRId64 "\n",
                    row,
                    column,
                    c[index],
                    expected);
            free(a);
            free(b);
            free(c);
            return 1;
        }
    }

    free(a);
    free(b);
    free(c);
    return 0;
}

static int check_loop_variants(void)
{
    const size_t m = 37;
    const size_t n = 73;
    const size_t k = 129;
    const size_t a_count = m * k;
    const size_t b_count = k * n;
    const size_t c_count = m * n;
    uint8_t *a = malloc(a_count * sizeof(*a));
    int8_t *b = malloc(b_count * sizeof(*b));
    int32_t *reference = calloc(c_count, sizeof(*reference));
    int32_t *ijk = calloc(c_count, sizeof(*ijk));
    int32_t *ikj = calloc(c_count, sizeof(*ikj));
    uint32_t state = UINT32_C(0x87654321);
    size_t index;
    int result = 0;

    if (a == NULL || b == NULL || reference == NULL || ijk == NULL || ikj == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(reference);
        free(ijk);
        free(ikj);
        return 1;
    }

    for (index = 0; index < a_count; ++index) {
        a[index] = (uint8_t)(next_random(&state) & UINT32_C(0xff));
    }
    for (index = 0; index < b_count; ++index) {
        b[index] = (int8_t)(next_random(&state) & UINT32_C(0xff));
    }

    gemm_u8s8s32(a, b, reference, m, n, k);
    gemm_u8s8s32_ijk(a, b, ijk, m, n, k);
    gemm_u8s8s32_ikj(a, b, ikj, m, n, k);

    for (index = 0; index < c_count; ++index) {
        if (reference[index] != ijk[index] || reference[index] != ikj[index]) {
            fprintf(stderr, "loop variant mismatch at output %zu\n", index);
            result = 1;
            break;
        }
    }

    free(a);
    free(b);
    free(reference);
    free(ijk);
    free(ikj);
    return result;
}

int main(void)
{
    if (check_non_square_case() != 0) {
        return EXIT_FAILURE;
    }
    if (check_loop_variants() != 0) {
        return EXIT_FAILURE;
    }

    puts("gemm_tests: all tests passed");
    return EXIT_SUCCESS;
}
