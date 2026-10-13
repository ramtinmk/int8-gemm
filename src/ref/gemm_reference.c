#include "gemm.h"

#include <stdint.h>

void gemm_u8s8s32(const uint8_t *a,
                  const int8_t *b,
                  int32_t *c,
                  size_t m,
                  size_t n,
                  size_t k)
{
    size_t i;

    for (i = 0; i < m; ++i) {
        size_t j;

        for (j = 0; j < n; ++j) {
            size_t x;
            int64_t accumulator = 0;

            for (x = 0; x < k; ++x) {
                const size_t a_index = i * k + x;
                const size_t b_index = x * n + j;

                accumulator += (int64_t)a[a_index] * (int64_t)b[b_index];
            }

            c[i * n + j] = (int32_t)accumulator;
        }
    }
}
