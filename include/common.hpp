// Common utilities shared by all variants (seq / openmp / mpi / opencl).
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Command-line arguments (identical for every program)
// ---------------------------------------------------------------------------
struct Args {
    int n = 500;           // rows of matrix1 / matrix3
    int k = 500;           // cols of matrix1 / rows of matrix2
    int p = 500;           // cols of matrix2 / matrix3
    int x = 10;            // matrix1 values are in [1, x]
    int threads = 0;       // OpenMP threads (0 = runtime default)
    uint64_t seed = 0;     // 0 = derive from current time
    bool print = false;    // print matrices
    bool verify = false;   // compare result with sequential reference
    bool csv = false;      // print a single CSV line instead of a report
    std::string kernel;    // OpenCL kernel file (opencl only)
};

Args parse_args(int argc, char** argv);

// ---------------------------------------------------------------------------
// Row-major matrix of ints
// ---------------------------------------------------------------------------
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> data;

    Matrix() = default;
    Matrix(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * static_cast<size_t>(c), 0) {}

    int*       data_ptr()                   { return data.data(); }
    const int* data_ptr() const             { return data.data(); }
    int*       row(size_t i)                { return data.data() + static_cast<size_t>(i) * cols; }
    const int* row(size_t i) const          { return data.data() + static_cast<size_t>(i) * cols; }
    int&       at(size_t i, size_t j)       { return data[static_cast<size_t>(i) * cols + j]; }
    int        at(size_t i, size_t j) const { return data[static_cast<size_t>(i) * cols + j]; }
};

// ---------------------------------------------------------------------------
// Random numbers
//
// matrix1: value of element (i, j) is a pure function of (seed, i, j).
//   This is thread-safe (no shared state, unlike rand()) and gives the same
//   matrix regardless of thread count or technology, so results are comparable.
// matrix2: generated sequentially with a single stateful generator.
// ---------------------------------------------------------------------------
inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Random integer in [1, x] for element with linear index idx
inline int rand_1_to_x(uint64_t seed, uint64_t idx, int x) {
    return 1 + static_cast<int>(splitmix64(seed ^ (idx * 0xD1B54A32D192ED03ull)) % static_cast<uint64_t>(x));
}

// Sequentially fills matrix2 with random 1 / -1 (shared by all variants)
void fill_matrix2_sequential(Matrix& m, uint64_t seed);

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------
inline double now_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Times {
    double fill1 = 0;  // matrix1 fill (parallel)
    double fill2 = 0;  // matrix2 fill (sequential)
    double mul = 0;    // multiplication (parallel)
    double total() const { return fill1 + fill2 + mul; }
};

// ---------------------------------------------------------------------------
// Output / verification
// ---------------------------------------------------------------------------
void print_matrix(const char* name, const Matrix& m);

// Recomputes matrix1 and matrix3 sequentially and compares. Returns true if equal.
bool verify_result(const Args& a, const Matrix& m1, const Matrix& m2, const Matrix& m3);

// Cheap partial check: recomputes `samples` elements of matrix3 (random positions plus
// the corners) from matrix1 and matrix2. Costs samples * k operations, so it can run
// on every benchmark run even for huge n. Prints the first mismatch to stderr.
bool spot_check(const Args& a, const Matrix& m1, const Matrix& m2, const Matrix& m3,
                int samples = 256);

// Prints the human-readable report or one CSV line.
// `workers` = threads / processes / GPU compute units, `note` = extra info (e.g. device).
void report(const Args& a, const char* tech, const char* variant, int workers,
            const Times& t, const std::string& note = "");

// Handles --print and --verify; returns process exit code (0 = ok).
int finish(const Args& a, const Matrix& m1, const Matrix& m2, const Matrix& m3);
