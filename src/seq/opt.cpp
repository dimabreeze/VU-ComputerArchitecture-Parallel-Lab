// Sequential optimized variant: same result as seq-base, one thread, no OpenMP.
// All speedup comes from the memory access pattern and the use of SIMD registers:
//   - loop order j0 -> k0 -> i: a GEMM_BK x GEMM_BJ tile of matrix2 is used for all n rows
//     of matrix1 before moving on, so it is read from RAM only once per tile
//   - packing: the tile of matrix2 is copied into a contiguous buffer, in panels
//     of 16 columns (zero padded), so the micro-kernel reads it strictly
//     sequentially (no page-stride access, see docs/tlb_cliff.md)
//   - micro-kernel 4 x 16: a 4-row x 16-column piece of matrix3 is kept in
//     registers for the whole k-block; per step it loads 16 ints of matrix2 and
//     4 ints of matrix1 for 64 multiply-adds
//   - compiled with -O3 -mcpu=native (see Makefile)
// The kernel itself is in include/gemm.hpp (shared with mpi-opt).
// Analysis and measurements: docs/seq_opt.md
#include "common.hpp"
#include "gemm.hpp"

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    Times t;

    // --- matrix1: random integers in [1, x] (same generator as all variants) ---
    double t0 = now_sec();
    for (int i = 0; i < a.n; ++i) {
        int* r = m1.row(i);
        const uint64_t base = static_cast<uint64_t>(i) * a.k;
        for (int j = 0; j < a.k; ++j) r[j] = rand_1_to_x(a.seed, base + j, a.x);
    }
    t.fill1 = now_sec() - t0;

    // --- matrix2: random 1 / -1 ---
    t0 = now_sec();
    fill_matrix2_sequential(m2, a.seed);
    t.fill2 = now_sec() - t0;

    // --- matrix3 = matrix1 * matrix2 (matrix3 is zero-initialized by Matrix) ---
    t0 = now_sec();
    gemm_add(a.n, 0, a.k, 0, a.p, m1.data_ptr(), a.k, m2.data_ptr(), a.p, m3.data_ptr(), a.p);
    t.mul = now_sec() - t0;

    report(a, "seq", "opt", 1, t);
    return finish(a, m1, m2, m3);
}
