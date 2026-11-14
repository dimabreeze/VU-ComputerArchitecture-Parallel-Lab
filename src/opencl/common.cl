// Code shared by all kernels in src/opencl/opt/*.cl (the host prepends this file).
//
// Contract of an opt kernel file:
//   __kernel void geometry(__global int* g)  writes {LX, LY, TM, TN}:
//       work-group size LX x LY (dimension 0 = columns, 1 = rows) and the TM x TN
//       block of matrix3 that one work-group computes
//   __kernel void matmul(A, B, C, n, k, p, row0)
//       computes rows row0 .. row0 + (number of row groups) * TM of C = A * B,
//       global size = (ceil(p / TN) * LX, ceil(rows / TM) * LY); must guard i < n, j < p
//
// Build options set by the host: -DUSE_MAD24 when all products fit in 24-bit
// operands (fast integer multiply-add on the GPU), plus anything from CL_BUILD_OPTS.

#ifdef USE_MAD24
#define MAD(a, b, c) mad24((a), (b), (c))
#else
#define MAD(a, b, c) ((a) * (b) + (c))
#endif

// Same generator as rand_1_to_x() in include/common.hpp (must stay identical!)
inline ulong splitmix64(ulong z) {
    z += 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

// matrix1[i][j] = random in [1, x]; global size = (k, n): dimension 0 = column j,
// so neighbouring work-items write neighbouring addresses (coalesced)
__kernel void fill_matrix1(__global int* m1, int n, int k, int x, ulong seed) {
    const int j = get_global_id(0);
    const int i = get_global_id(1);
    if (i >= n || j >= k) return;
    const ulong idx = (ulong)i * (ulong)k + (ulong)j;
    m1[idx] = 1 + (int)(splitmix64(seed ^ (idx * 0xD1B54A32D192ED03UL)) % (ulong)x);
}
