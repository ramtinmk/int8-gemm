#ifndef TINY_INT8_GEMM_H
#define TINY_INT8_GEMM_H

#include <stddef.h>
#include <stdint.h>

/*
 * Computes C = A * B for row-major matrices:
 *
 *   A: M x K, unsigned 8-bit elements
 *   B: K x N, signed 8-bit elements
 *   C: M x N, signed 32-bit elements
 *
 * The caller owns all buffers, which must contain at least the indicated
 * number of elements and must not overlap. The result is defined for inputs
 * whose mathematical dot products fit in int32_t.
 */
void gemm_u8s8s32(const uint8_t *a,
                  const int8_t *b,
                  int32_t *c,
                  size_t m,
                  size_t n,
                  size_t k);

/* Scalar loop-order variants used for phase-2 memory-access measurements. */
void gemm_u8s8s32_ijk(const uint8_t *a,
                      const int8_t *b,
                      int32_t *c,
                      size_t m,
                      size_t n,
                      size_t k);

void gemm_u8s8s32_ikj(const uint8_t *a,
                      const int8_t *b,
                      int32_t *c,
                      size_t m,
                      size_t n,
                      size_t k);

#endif
