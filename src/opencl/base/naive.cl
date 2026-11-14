// OpenCL kernels: one work-item per matrix element.

// Same generator as rand_1_to_x() in include/common.hpp (must stay identical!)
inline ulong splitmix64(ulong z) {
    z += 0x9E3779B97F4A7C15UL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}

// matrix1[i][j] = random in [1, x]; global size = (n, k)
__kernel void fill_matrix1(__global int* m1, int n, int k, int x, ulong seed) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= n || j >= k) return;
    const ulong idx = (ulong)i * (ulong)k + (ulong)j;
    m1[idx] = 1 + (int)(splitmix64(seed ^ (idx * 0xD1B54A32D192ED03UL)) % (ulong)x);
}

// matrix3 = matrix1 * matrix2; global size = (n, p)
__kernel void matmul(__global const int* m1, __global const int* m2, __global int* m3,
                     int n, int k, int p) {
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    if (i >= n || j >= p) return;
    int sum = 0;
    for (int z = 0; z < k; ++z) sum += m1[i * k + z] * m2[z * p + j];
    m3[i * p + j] = sum;
}
