#define _POSIX_C_SOURCE 199309L

#include "gemm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end)
{
    const time_t seconds = end->tv_sec - start->tv_sec;
    const long nanoseconds = end->tv_nsec - start->tv_nsec;

    return (double)seconds + (double)nanoseconds / 1000000000.0;
}

int main(void)
{
    const size_t m = 256;
    const size_t n = 256;
    const size_t k = 256;
    const size_t iterations = 5;
    const size_t a_count = m * k;
    const size_t b_count = k * n;
    const size_t c_count = m * n;
    uint8_t *a = malloc(a_count * sizeof(*a));
    int8_t *b = malloc(b_count * sizeof(*b));
    int32_t *c = calloc(c_count, sizeof(*c));
    struct timespec start;
    struct timespec end;
    double seconds;
    size_t iteration;

    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(c);
        return EXIT_FAILURE;
    }

    for (iteration = 0; iteration < a_count; ++iteration) {
        a[iteration] = (uint8_t)(iteration & 0xffU);
    }
    for (iteration = 0; iteration < b_count; ++iteration) {
        b[iteration] = (int8_t)(iteration & 0x7fU);
    }

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        fprintf(stderr, "clock_gettime failed\n");
        free(a);
        free(b);
        free(c);
        return EXIT_FAILURE;
    }
    for (iteration = 0; iteration < iterations; ++iteration) {
        gemm_u8s8s32(a, b, c, m, n, k);
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        fprintf(stderr, "clock_gettime failed\n");
        free(a);
        free(b);
        free(c);
        return EXIT_FAILURE;
    }

    seconds = elapsed_seconds(&start, &end);
    printf("M=%zu N=%zu K=%zu iterations=%zu seconds=%.6f gops=%.3f\n",
           m,
           n,
           k,
           iterations,
           seconds,
           (2.0 * (double)m * (double)n * (double)k * (double)iterations) /
               (seconds * 1000000000.0));

    free(a);
    free(b);
    free(c);
    return EXIT_SUCCESS;
}
