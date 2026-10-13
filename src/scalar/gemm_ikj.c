#include "gemm.h"

#include <stdint.h>

void gemm_u8s8s32_ikj(const uint8_t *a,
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
            c[i * n + j] = 0;
        }

        for (size_t x = 0; x < k; ++x) {
            const int64_t a_value = (int64_t)a[i * k + x];

            for (j = 0; j < n; ++j) {
                const size_t c_index = i * n + j;
                const size_t b_index = x * n + j;
                const int64_t product = a_value * (int64_t)b[b_index];

                c[c_index] = (int32_t)((int64_t)c[c_index] + product);
            }
        }
    }
}
