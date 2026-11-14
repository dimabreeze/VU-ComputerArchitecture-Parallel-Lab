// Fast single-threaded integer matrix multiplication kernel (shared by seq-opt, mpi-opt).
//
//   gemm_add(rows, k0, k1, j0, j1, A, lda, B, ldb, C, ldc):
//       C[0:rows, j0:j1] += A[0:rows, k0:k1] * B[k0:k1, j0:j1]
//   All matrices are row-major; lda / ldb / ldc are row lengths (in ints).
//   A row index r of A / C is local (0..rows-1), k and j indices are global, so a
//   caller can multiply its own block of rows by any k-range of B (MPI pipeline).
//
// Techniques (details and measurements: docs/seq_opt.md):
//   - blocking  j0 (GEMM_BJ) -> k0 (GEMM_BK) -> i0 (GEMM_BI) -> 16-column panel -> MR rows
//   - packing: each GEMM_BK x GEMM_BJ tile of B is copied into a contiguous buffer as
//     panels of 16 columns (zero padded), so B is read strictly sequentially
//   - register micro-kernel GEMM_MR x 16: the C tile stays in NEON registers for the
//     whole k-block
// Compile with -O3 -mcpu=native for full speed (see Makefile).
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#ifndef GEMM_MR
#define GEMM_MR 4     // rows of C in the micro-kernel
#endif
#ifndef GEMM_BI
#define GEMM_BI 64    // block of A / C rows (multiple of GEMM_MR)
#endif
#ifndef GEMM_BK
#define GEMM_BK 512   // block of the inner (k) dimension
#endif
#ifndef GEMM_BJ
#define GEMM_BJ 512   // block of B / C columns (multiple of 16)
#endif

namespace gemm_detail {

constexpr int NR = 16;  // columns of C in the micro-kernel (4 NEON registers of 4 ints)

// Copies B[k0:k1, j0:j1] into `buf` as panels of NR columns: panel q holds columns
// j0 + q*NR .. +NR, row by row. Columns past j1 are zero-filled, so the micro-kernel
// never needs a column tail.
inline void pack_b(const int* B, int ldb, int k0, int k1, int j0, int j1, int* buf) {
    const int kb = k1 - k0;
    for (int jq = j0; jq < j1; jq += NR) {
        const int w = std::min(NR, j1 - jq);
        int* panel = buf + static_cast<size_t>((jq - j0) / NR) * kb * NR;
        for (int z = k0; z < k1; ++z) {
            const int* src = B + static_cast<size_t>(z) * ldb + jq;
            int* dst = panel + static_cast<size_t>(z - k0) * NR;
            int q = 0;
            for (; q < w; ++q) dst[q] = src[q];
            for (; q < NR; ++q) dst[q] = 0;
        }
    }
}

// C[i:i+R, j:j+w] += A[i:i+R, k0:k1] * panel,  R <= GEMM_MR, w <= NR
template <int R>
inline void micro(const int* A, int lda, int* C, int ldc, int i, int j, int w, int k0, int k1,
                  const int* __restrict panel) {
    const int* a[R];
    for (int r = 0; r < R; ++r) a[r] = A + static_cast<size_t>(i + r) * lda;
    int acc[R][NR] = {};
    for (int z = k0; z < k1; ++z) {
        const int* __restrict b = panel + static_cast<size_t>(z - k0) * NR;
        for (int r = 0; r < R; ++r) {
            const int v = a[r][z];
            for (int q = 0; q < NR; ++q) acc[r][q] += v * b[q];
        }
    }
    // acc is indexed only with compile-time constants, so the compiler keeps it in
    // registers; a variable index (q < w) here would force acc to the stack and add
    // a store of all accumulators on every z step (2x slower).
    for (int r = 0; r < R; ++r) {
        int* c = C + static_cast<size_t>(i + r) * ldc + j;
        if (w == NR) {
            for (int q = 0; q < NR; ++q) c[q] += acc[r][q];
        } else {
            int tmp[NR];
            for (int q = 0; q < NR; ++q) tmp[q] = acc[r][q];
            for (int q = 0; q < w; ++q) c[q] += tmp[q];
        }
    }
}

}  // namespace gemm_detail

// C[0:rows, j0:j1] += A[0:rows, k0:k1] * B[k0:k1, j0:j1]
inline void gemm_add(int rows, int k0, int k1, int j0, int j1,
                     const int* A, int lda, const int* B, int ldb, int* C, int ldc) {
    using namespace gemm_detail;
    static thread_local std::vector<int> buf(static_cast<size_t>(GEMM_BK) * GEMM_BJ);
    for (int jb = j0; jb < j1; jb += GEMM_BJ) {
        const int je = std::min(jb + GEMM_BJ, j1);
        for (int kb = k0; kb < k1; kb += GEMM_BK) {
            const int ke = std::min(kb + GEMM_BK, k1);
            pack_b(B, ldb, kb, ke, jb, je, buf.data());
            for (int i0 = 0; i0 < rows; i0 += GEMM_BI) {
                const int i1 = std::min(i0 + GEMM_BI, rows);
                for (int jq = jb; jq < je; jq += NR) {
                    const int w = std::min(NR, je - jq);
                    const int* panel =
                        buf.data() + static_cast<size_t>((jq - jb) / NR) * (ke - kb) * NR;
                    int i = i0;
                    for (; i + GEMM_MR <= i1; i += GEMM_MR)
                        micro<GEMM_MR>(A, lda, C, ldc, i, jq, w, kb, ke, panel);
                    for (; i < i1; ++i) micro<1>(A, lda, C, ldc, i, jq, w, kb, ke, panel);
                }
            }
        }
    }
}
