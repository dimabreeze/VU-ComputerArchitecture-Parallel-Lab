// Step 2: tiling in local (on-chip) memory.
// A work-group of TS x TS work-items computes a TS x TS block of matrix3. For each
// step of TS along k, the group loads a TS x TS tile of matrix1 and of matrix2 into
// __local memory (every work-item loads one element of each), synchronizes, and
// then every loaded element is used TS times from fast local memory.
// Global memory traffic drops by a factor of TS compared to coalesced.cl.

#ifndef TS
#define TS 16
#endif

__kernel void geometry(__global int* g) {
    g[0] = TS; g[1] = TS; g[2] = TS; g[3] = TS;
}

__kernel __attribute__((reqd_work_group_size(TS, TS, 1)))
void matmul(__global const int* A, __global const int* B, __global int* C,
            int n, int k, int p, int row0) {
    const int lx = get_local_id(0);  // column inside the tile
    const int ly = get_local_id(1);  // row inside the tile
    const int j = get_group_id(0) * TS + lx;
    const int i = row0 + get_group_id(1) * TS + ly;

    __local int As[TS][TS];
    __local int Bs[TS][TS];

    int sum = 0;
    for (int t = 0; t < k; t += TS) {
        // out-of-range elements are loaded as 0, so edges need no special case
        As[ly][lx] = (i < n && t + lx < k) ? A[(size_t)i * k + t + lx] : 0;
        Bs[ly][lx] = (t + ly < k && j < p) ? B[(size_t)(t + ly) * p + j] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int z = 0; z < TS; ++z) sum = MAD(As[ly][z], Bs[z][lx], sum);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < n && j < p) C[(size_t)i * p + j] = sum;
}
