// OpenMP optimized variant (starting point for experiments).
// Differences from base:
//   - i-k-j loop order: inner loop walks rows of matrix2 and matrix3
//     contiguously (cache friendly, vectorizable) instead of striding columns
//   - plain static schedule: one contiguous block of rows per thread
#include <omp.h>

#include "common.hpp"

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.threads > 0) omp_set_num_threads(a.threads);

    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    Times t;
    int num_threads = 0;

    double t0 = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();

#pragma omp for schedule(static)
        for (int i = 0; i < a.n; ++i) {
            int* r = m1.row(i);
            const uint64_t base = static_cast<uint64_t>(i) * a.k;
            for (int j = 0; j < a.k; ++j) r[j] = rand_1_to_x(a.seed, base + j, a.x);
        }
    }
    t.fill1 = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    fill_matrix2_sequential(m2, a.seed);
    t.fill2 = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < a.n; ++i) {
        int* c = m3.row(i);
        const int* ai = m1.row(i);
        for (int j = 0; j < a.p; ++j) c[j] = 0;
        for (int z = 0; z < a.k; ++z) {
            const int v = ai[z];
            const int* b = m2.row(z);
#pragma omp simd
            for (int j = 0; j < a.p; ++j) c[j] += v * b[j];
        }
    }
    t.mul = omp_get_wtime() - t0;

    report(a, "openmp", "opt", num_threads, t);
    return finish(a, m1, m2, m3);
}
