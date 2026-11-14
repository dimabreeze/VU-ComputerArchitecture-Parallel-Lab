// OpenMP optimized variant #2 (builds on opt.cpp).
// Differences from opt:
//   - cache blocking (tiling): rows of matrix3 are processed in blocks of BI rows,
//     the k and p dimensions in blocks of BK x BJ, so the current tile of matrix2
//     stays in cache and is reused by all BI rows instead of being re-read
//     from RAM for every row (opt is memory-bandwidth bound for large n)
//   - register blocking (micro-kernel): a 4 x 16 tile of matrix3 is kept in
//     registers for the whole k-block, every loaded matrix2 element is used 4 times
//   - dynamic schedule over row blocks: balances load between fast (P) and
//     slow (E) cores when more threads than P-cores are used
//   - compiled with -O3 -mcpu=native (see Makefile)
// Analysis, block size tuning and measurements: docs/openmp_opt2.md
#include <algorithm>

#include <omp.h>

#include "common.hpp"

#ifndef BI
#define BI 128   // rows of matrix3 per task
#endif
#ifndef BK
#define BK 256   // block of the inner (k) dimension
#endif
#ifndef BJ
#define BJ 128   // block of matrix3 / matrix2 columns
#endif

// Micro-kernel: 4 x MJ tile of matrix3 kept in registers (acc) for the whole z loop,
// so per z there are only MJ loads of matrix2 + 4 loads of matrix1 for 4*MJ multiply-adds.
constexpr int MJ = 16;
static inline void micro4x16(const Matrix& m1, const Matrix& m2, Matrix& m3, int i, int j,
                             int k0, int k1) {
    const int* a0 = m1.row(i);
    const int* a1 = m1.row(i + 1);
    const int* a2 = m1.row(i + 2);
    const int* a3 = m1.row(i + 3);
    int acc[4][MJ] = {};
    for (int z = k0; z < k1; ++z) {
        const int* b = m2.row(z) + j;
        const int v[4] = {a0[z], a1[z], a2[z], a3[z]};
        for (int r = 0; r < 4; ++r)
            for (int q = 0; q < MJ; ++q) acc[r][q] += v[r] * b[q];
    }
    for (int r = 0; r < 4; ++r) {
        int* c = m3.row(i + r) + j;
        for (int q = 0; q < MJ; ++q) c[q] += acc[r][q];
    }
}

// c[r][j] += sum over z in [k0, k1) of a[r][z] * b[z][j],  r = 0..3,  j in [j0, j1)
// (used for the column tail narrower than MJ)
static inline void kernel4(const Matrix& m1, const Matrix& m2, Matrix& m3, int i,
                           int k0, int k1, int j0, int j1) {
    int* __restrict c0 = m3.row(i);
    int* __restrict c1 = m3.row(i + 1);
    int* __restrict c2 = m3.row(i + 2);
    int* __restrict c3 = m3.row(i + 3);
    const int* a0 = m1.row(i);
    const int* a1 = m1.row(i + 1);
    const int* a2 = m1.row(i + 2);
    const int* a3 = m1.row(i + 3);
    for (int z = k0; z < k1; ++z) {
        const int* __restrict b = m2.row(z);
        const int v0 = a0[z], v1 = a1[z], v2 = a2[z], v3 = a3[z];
#pragma omp simd
        for (int j = j0; j < j1; ++j) {
            const int bj = b[j];
            c0[j] += v0 * bj;
            c1[j] += v1 * bj;
            c2[j] += v2 * bj;
            c3[j] += v3 * bj;
        }
    }
}

// Same for a single row (tail when the block has fewer than 4 rows left)
static inline void kernel1(const Matrix& m1, const Matrix& m2, Matrix& m3, int i,
                           int k0, int k1, int j0, int j1) {
    int* __restrict c = m3.row(i);
    const int* a = m1.row(i);
    for (int z = k0; z < k1; ++z) {
        const int* __restrict b = m2.row(z);
        const int v = a[z];
#pragma omp simd
        for (int j = j0; j < j1; ++j) c[j] += v * b[j];
    }
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.threads > 0) omp_set_num_threads(a.threads);

    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    Times t;
    int num_threads = 0;

    // --- matrix1: random integers in [1, x], parallel ---
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

    // --- matrix2: random 1 / -1, sequential ---
    t0 = omp_get_wtime();
    fill_matrix2_sequential(m2, a.seed);
    t.fill2 = omp_get_wtime() - t0;

    // --- matrix3 = matrix1 * matrix2, parallel, blocked ---
    // matrix3 is zero-initialized by the Matrix constructor.
    t0 = omp_get_wtime();
#pragma omp parallel for schedule(dynamic, 1)
    for (int i0 = 0; i0 < a.n; i0 += BI) {
        const int i1 = std::min(i0 + BI, a.n);
        for (int k0 = 0; k0 < a.k; k0 += BK) {
            const int k1 = std::min(k0 + BK, a.k);
            for (int j0 = 0; j0 < a.p; j0 += BJ) {
                const int j1 = std::min(j0 + BJ, a.p);
                int i = i0;
                for (; i + 4 <= i1; i += 4) {
                    int j = j0;
                    for (; j + MJ <= j1; j += MJ) micro4x16(m1, m2, m3, i, j, k0, k1);
                    if (j < j1) kernel4(m1, m2, m3, i, k0, k1, j, j1);
                }
                for (; i < i1; ++i) kernel1(m1, m2, m3, i, k0, k1, j0, j1);
            }
        }
    }
    t.mul = omp_get_wtime() - t0;

    report(a, "openmp", "opt2", num_threads, t);
    return finish(a, m1, m2, m3);
}
