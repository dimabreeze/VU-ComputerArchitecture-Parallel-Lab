// Step 3: tiling in local memory + register blocking.
// A work-group of LX x LY work-items computes a TM x TN block of matrix3; every
// work-item computes WM x WN elements (TM = LY * WM, TN = LX * WN) and keeps them in
// registers. Per step of TK along k the group loads a TM x TK tile of matrix1 and a
// TK x TN tile of matrix2 into local memory; then each work-item reads WM + WN values
// from local memory per z and does WM * WN multiply-adds (64 for 8 x 8), instead of
// 2 reads per multiply-add in tiled.cl. This is the GPU counterpart of the CPU
// micro-kernel in include/gemm.hpp.
//
// Work-item (lx, ly) owns rows ly + LY*r and columns lx + LX*c (strided), so that
// neighbouring work-items access neighbouring columns of the local tile of matrix2
// (no bank conflicts) and neighbouring addresses in global memory.

#ifndef WM
#define WM 8     // rows of matrix3 per work-item
#endif
#ifndef WN
#define WN 8     // columns of matrix3 per work-item
#endif
#ifndef LX
#define LX 16
#endif
#ifndef LY
#define LY 16
#endif
#ifndef TK
#define TK 16    // tile depth along k
#endif
#define TM (LY * WM)
#define TN (LX * WN)
#define NT (LX * LY)  // work-items per group

__kernel void geometry(__global int* g) {
    g[0] = LX; g[1] = LY; g[2] = TM; g[3] = TN;
}

__kernel __attribute__((reqd_work_group_size(LX, LY, 1)))
void matmul(__global const int* A, __global const int* B, __global int* C,
            int n, int k, int p, int row0) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int tid = ly * LX + lx;
    const int i0 = row0 + get_group_id(1) * TM;  // first row of the block
    const int j0 = get_group_id(0) * TN;         // first column of the block

    __local int As[TM][TK + 1];  // +1: avoid bank conflicts when reading a column
    __local int Bs[TK][TN];

    int acc[WM][WN];
    for (int r = 0; r < WM; ++r)
        for (int c = 0; c < WN; ++c) acc[r][c] = 0;

    for (int t = 0; t < k; t += TK) {
        // cooperative loads, out-of-range elements as 0
        for (int idx = tid; idx < TM * TK; idx += NT) {
            const int r = idx / TK, c = idx % TK;
            const int gi = i0 + r, gz = t + c;
            As[r][c] = (gi < n && gz < k) ? A[(size_t)gi * k + gz] : 0;
        }
        for (int idx = tid; idx < TK * TN; idx += NT) {
            const int r = idx / TN, c = idx % TN;
            const int gz = t + r, gj = j0 + c;
            Bs[r][c] = (gz < k && gj < p) ? B[(size_t)gz * p + gj] : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int z = 0; z < TK; ++z) {
            int a[WM], b[WN];
            for (int r = 0; r < WM; ++r) a[r] = As[ly + LY * r][z];
            for (int c = 0; c < WN; ++c) b[c] = Bs[z][lx + LX * c];
            for (int r = 0; r < WM; ++r)
                for (int c = 0; c < WN; ++c) acc[r][c] = MAD(a[r], b[c], acc[r][c]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int r = 0; r < WM; ++r) {
        const int i = i0 + ly + LY * r;
        if (i >= n) continue;
        for (int c = 0; c < WN; ++c) {
            const int j = j0 + lx + LX * c;
            if (j < p) C[(size_t)i * p + j] = acc[r][c];
        }
    }
}
