// OpenMP variant strictly following the task:
//   matrix1 fill    -> parallel (chosen number of threads)
//   matrix2 fill    -> sequential
//   multiplication  -> parallel (chosen number of threads)
// Based on the course template ("OpenMP matrix training.cpp"), with fixes:
//   - thread-safe random generator instead of rand()
//   - timer declared outside the parallel region, timed per phase
//   - sizes / threads from command line, matrices on the heap
#include <omp.h>

#include "common.hpp"

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.threads > 0) omp_set_num_threads(a.threads);

    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    Times t;
    int num_threads = 0;
    const int chunk = 4;

    // --- matrix1: random integers in [1, x], parallel ---
    double t0 = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();

#pragma omp for schedule(static, chunk)
        for (int i = 0; i < a.n; ++i)
            for (int j = 0; j < a.k; ++j)
                m1.at(i, j) = rand_1_to_x(a.seed, static_cast<uint64_t>(i) * a.k + j, a.x);
    }
    t.fill1 = omp_get_wtime() - t0;

    // --- matrix2: random 1 / -1, sequential ---
    t0 = omp_get_wtime();
    fill_matrix2_sequential(m2, a.seed);
    t.fill2 = omp_get_wtime() - t0;

    // --- matrix3 = matrix1 * matrix2, parallel ---
    t0 = omp_get_wtime();
#pragma omp parallel for schedule(static, chunk)
    for (int i = 0; i < a.n; ++i)
        for (int j = 0; j < a.p; ++j) {
            int sum = 0;
            for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
            m3.at(i, j) = sum;
        }
    t.mul = omp_get_wtime() - t0;

    report(a, "openmp", "base", num_threads, t);
    return finish(a, m1, m2, m3);
}
