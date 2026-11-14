// Sequential reference: same steps as the task, no parallelism.
// Used as the baseline for speedup calculation.
#include "common.hpp"

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    Times t;

    double t0 = now_sec();
    for (int i = 0; i < a.n; ++i)
        for (int j = 0; j < a.k; ++j)
            m1.at(i, j) = rand_1_to_x(a.seed, static_cast<uint64_t>(i) * a.k + j, a.x);
    t.fill1 = now_sec() - t0;

    t0 = now_sec();
    fill_matrix2_sequential(m2, a.seed);
    t.fill2 = now_sec() - t0;

    t0 = now_sec();
    for (int i = 0; i < a.n; ++i)
        for (int j = 0; j < a.p; ++j) {
            int sum = 0;
            for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
            m3.at(i, j) = sum;
        }
    t.mul = now_sec() - t0;

    report(a, "seq", "base", 1, t);
    return finish(a, m1, m2, m3);
}
