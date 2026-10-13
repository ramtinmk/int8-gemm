#include "gemm.h"

#include <stdint.h>

void gemm_u8s8s32_ijk(const uint8_t *a,
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
                accumulator += (int64_t)a[i * k + x] *
                               (int64_t)b[x * n + j];
            }
            c[i * n + j] = (int32_t)accumulator;
        }
    }
}
