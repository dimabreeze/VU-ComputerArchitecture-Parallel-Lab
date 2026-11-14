#include "common.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -n N         rows of matrix1 / matrix3        (default 500)\n"
        "  -k K         cols of matrix1 / rows matrix2   (default 500)\n"
        "  -p P         cols of matrix2 / matrix3        (default 500)\n"
        "  -s S         shortcut: n = k = p = S\n"
        "  -x X         matrix1 values in [1, X]         (default 10)\n"
        "  -t T         OpenMP threads                   (default: runtime)\n"
        "  --seed S     random seed                      (default: time)\n"
        "  --kernel F   OpenCL kernel file\n"
        "  --print      print matrices\n"
        "  --verify     check result against sequential reference\n"
        "  --csv        print one CSV line\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { usage(argv[0]); std::exit(2); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const char* s = argv[i];
        if      (!std::strcmp(s, "-n")) a.n = std::atoi(need(i));
        else if (!std::strcmp(s, "-k")) a.k = std::atoi(need(i));
        else if (!std::strcmp(s, "-p")) a.p = std::atoi(need(i));
        else if (!std::strcmp(s, "-s")) a.n = a.k = a.p = std::atoi(need(i));
        else if (!std::strcmp(s, "-x")) a.x = std::atoi(need(i));
        else if (!std::strcmp(s, "-t")) a.threads = std::atoi(need(i));
        else if (!std::strcmp(s, "--seed"))   a.seed = std::strtoull(need(i), nullptr, 10);
        else if (!std::strcmp(s, "--kernel")) a.kernel = need(i);
        else if (!std::strcmp(s, "--print"))  a.print = true;
        else if (!std::strcmp(s, "--verify")) a.verify = true;
        else if (!std::strcmp(s, "--csv"))    a.csv = true;
        else if (!std::strcmp(s, "-h") || !std::strcmp(s, "--help")) { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Unknown option: %s\n", s); usage(argv[0]); std::exit(2); }
    }
    if (a.n <= 0 || a.k <= 0 || a.p <= 0 || a.x <= 0 || a.threads < 0) {
        std::fprintf(stderr, "Sizes, x must be > 0 and threads >= 0\n");
        std::exit(2);
    }
    if (a.seed == 0) a.seed = static_cast<uint64_t>(std::time(nullptr));
    return a;
}

void fill_matrix2_sequential(Matrix& m, uint64_t seed) {
    std::mt19937_64 gen(seed + 1);
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            m.at(i, j) = (gen() & 1) ? 1 : -1;
}

void print_matrix(const char* name, const Matrix& m) {
    std::printf("\n%s (%dx%d):\n", name, m.rows, m.cols);
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) std::printf("%6d", m.at(i, j));
        std::printf("\n");
    }
}

bool verify_result(const Args& a, const Matrix& m1, const Matrix& m2, const Matrix& m3) {
    // matrix1 must match the deterministic generator
    for (int i = 0; i < a.n; ++i)
        for (int j = 0; j < a.k; ++j)
            if (m1.at(i, j) != rand_1_to_x(a.seed, static_cast<uint64_t>(i) * a.k + j, a.x)) {
                std::fprintf(stderr, "VERIFY FAILED: matrix1[%d][%d]\n", i, j);
                return false;
            }
    // matrix3 must equal matrix1 * matrix2 (simple sequential i-k-j product)
    std::vector<int> rowbuf(a.p);
    for (int i = 0; i < a.n; ++i) {
        std::fill(rowbuf.begin(), rowbuf.end(), 0);
        for (int z = 0; z < a.k; ++z) {
            const int v = m1.at(i, z);
            const int* b = m2.row(z);
            for (int j = 0; j < a.p; ++j) rowbuf[j] += v * b[j];
        }
        for (int j = 0; j < a.p; ++j)
            if (m3.at(i, j) != rowbuf[j]) {
                std::fprintf(stderr, "VERIFY FAILED: matrix3[%d][%d] = %d, expected %d\n",
                             i, j, m3.at(i, j), rowbuf[j]);
                return false;
            }
    }
    return true;
}

bool spot_check(const Args& a, const Matrix& m1, const Matrix& m2, const Matrix& m3,
                int samples) {
    auto check = [&](int i, int j) {
        int sum = 0;
        for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
        if (m3.at(i, j) != sum) {
            std::fprintf(stderr, "SPOT CHECK FAILED: matrix3[%d][%d] = %d, expected %d\n", i, j,
                         m3.at(i, j), sum);
            return false;
        }
        return true;
    };
    if (!check(0, 0) || !check(a.n - 1, a.p - 1) || !check(0, a.p - 1) || !check(a.n - 1, 0))
        return false;
    for (int s = 0; s < samples; ++s) {
        const uint64_t r = splitmix64(a.seed + 0x5151 + static_cast<uint64_t>(s));
        const int i = static_cast<int>((r >> 32) % static_cast<uint64_t>(a.n));
        const int j = static_cast<int>((r & 0xFFFFFFFFu) % static_cast<uint64_t>(a.p));
        if (!check(i, j)) return false;
    }
    return true;
}

void report(const Args& a, const char* tech, const char* variant, int workers,
            const Times& t, const std::string& note) {
    if (a.csv) {
        // tech,variant,n,k,p,workers,fill1,fill2,mul,total
        std::printf("%s,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n", tech, variant, a.n, a.k, a.p,
                    workers, t.fill1, t.fill2, t.mul, t.total());
        return;
    }
    std::printf("=== %s / %s ===\n", tech, variant);
    std::printf("Matrix1: %dx%d, Matrix2: %dx%d, Matrix3: %dx%d, x = %d, seed = %llu\n",
                a.n, a.k, a.k, a.p, a.n, a.p, a.x, static_cast<unsigned long long>(a.seed));
    std::printf("Workers: %d%s%s\n", workers, note.empty() ? "" : "  ", note.c_str());
    std::printf("  fill matrix1 (parallel)   : %10.6f s\n", t.fill1);
    std::printf("  fill matrix2 (sequential) : %10.6f s\n", t.fill2);
    std::printf("  multiply     (parallel)   : %10.6f s\n", t.mul);
    std::printf("  total                     : %10.6f s\n", t.total());
}

int finish(const Args& a, const Matrix& m1, const Matrix& m2, const Matrix& m3) {
    if (a.print) {
        print_matrix("Matrix1", m1);
        print_matrix("Matrix2", m2);
        print_matrix("Matrix3", m3);
    }
    if (a.verify) {
        const bool ok = verify_result(a, m1, m2, m3);
        if (!a.csv) std::printf("Verify: %s\n", ok ? "OK" : "FAILED");
        return ok ? 0 : 1;
    }
    return 0;
}
