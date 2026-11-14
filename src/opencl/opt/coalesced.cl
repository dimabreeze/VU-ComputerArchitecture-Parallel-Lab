// Step 1: one work-item per element of matrix3 (like naive.cl), but
//   - dimension 0 of the NDRange is the column j: neighbouring work-items read
//     neighbouring elements of matrix2 (one coalesced memory transaction) and the
//     same element of matrix1 (broadcast); naive.cl had i in dimension 0, so every
//     work-item read its own cache line of matrix1 (stride k)
//   - mad24 integer multiply-add (when the host allows it, see common.cl)
//   - fixed work-group size LX x LY

#ifndef LX
#define LX 16
#endif
#ifndef LY
#define LY 16
#endif

__kernel void geometry(__global int* g) {
    g[0] = LX; g[1] = LY; g[2] = LY; g[3] = LX;  // TM = LY, TN = LX
}

__kernel __attribute__((reqd_work_group_size(LX, LY, 1)))
void matmul(__global const int* A, __global const int* B, __global int* C,
            int n, int k, int p, int row0) {
    const int j = get_global_id(0);
    const int i = row0 + get_global_id(1);
    if (i >= n || j >= p) return;
    __global const int* a = A + (size_t)i * k;
    __global const int* b = B + j;
    int sum = 0;
    for (int z = 0; z < k; ++z) sum = MAD(a[z], b[(size_t)z * p], sum);
    C[(size_t)i * p + j] = sum;
}
