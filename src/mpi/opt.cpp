// MPI optimized variant (builds on base.cpp, same task semantics):
//   matrix1 fill    -> parallel: each process fills its own block of rows
//   matrix2 fill    -> sequential on rank 0
//   multiplication  -> parallel: each process computes its block of rows
// Differences from task:
//   - fast kernel: include/gemm.hpp (blocking + packing + register micro-kernel,
//     same as seq-opt) instead of the i-j-z loop (which hits the TLB limit,
//     see docs/tlb_cliff.md)
//   - block row distribution instead of cyclic: every process owns rows
//     [offset, offset + rows); only these rows of matrix1 / matrix3 are stored,
//     matrix3 is collected with one MPI_Gatherv instead of n Send/Recv messages
//   - optional pipelined broadcast of matrix2 (-DKC=<rows>): it is sent in chunks
//     of KC rows with non-blocking MPI_Ibcast and a process multiplies chunk c
//     while chunks c+1... are being transferred. OFF by default (KC = 0 = one
//     chunk with all k rows): on a single shared-memory machine the broadcast is
//     cheap (~15 ms for 64 MB) and does not really overlap with computation, so
//     the pipeline was measured to be 5-20% slower (docs/mpi_opt.md).
//     The broadcast is timed in the multiplication phase in both modes (in
//     base.cpp it was part of the matrix2 phase).
//   - compiled with -O3 -mcpu=native (see Makefile)
// Analysis and measurements: docs/mpi_opt.md
#include <algorithm>
#include <vector>

#include <mpi.h>

#include "common.hpp"
#include "gemm.hpp"

#ifndef KC
#define KC 0     // rows of matrix2 per broadcast chunk; 0 = no pipeline (one chunk of k rows)
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Args a = parse_args(argc, argv);
    // all ranks must use the same seed (time-based seed may differ between ranks)
    MPI_Bcast(&a.seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    // --- block distribution of rows: rank r owns rows [offs[r], offs[r] + rows[r]) ---
    std::vector<int> rows(size), offs(size);
    for (int r = 0, off = 0; r < size; ++r) {
        rows[r] = a.n / size + (r < a.n % size ? 1 : 0);
        offs[r] = off;
        off += rows[r];
    }
    const int my_rows = rows[rank];
    const int my_off = offs[rank];

    // Rank 0 owns the first block, so it works directly in the full matrices
    // (needed anyway for the result and for --print / --verify); other ranks
    // store only their own rows of matrix1 and matrix3.
    Matrix m1(rank == 0 ? a.n : my_rows, a.k);
    Matrix m3(rank == 0 ? a.n : my_rows, a.p);
    Matrix m2(a.k, a.p);  // every process needs the whole matrix2
    Times t;

    // --- matrix1: each process fills its own rows (parallel) ---
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < my_rows; ++i) {
        int* r = m1.row(i);
        const uint64_t base = static_cast<uint64_t>(my_off + i) * a.k;  // global row index
        for (int j = 0; j < a.k; ++j) r[j] = rand_1_to_x(a.seed, base + j, a.x);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t.fill1 = MPI_Wtime() - t0;

    // --- matrix2: sequential on rank 0 (broadcast happens in the next phase) ---
    t0 = MPI_Wtime();
    if (rank == 0) fill_matrix2_sequential(m2, a.seed);
    MPI_Barrier(MPI_COMM_WORLD);
    t.fill2 = MPI_Wtime() - t0;

    // --- multiplication with pipelined broadcast of matrix2 ---
    t0 = MPI_Wtime();
    const int kc = (KC > 0) ? KC : a.k;  // chunk size in rows of matrix2
    const int chunks = (a.k + kc - 1) / kc;
    std::vector<MPI_Request> req(chunks);
    for (int c = 0; c < chunks; ++c) {  // post all chunks; MPI transfers them in order
        const int k0 = c * kc, k1 = std::min(k0 + kc, a.k);
        MPI_Ibcast(m2.row(k0), (k1 - k0) * a.p, MPI_INT, 0, MPI_COMM_WORLD, &req[c]);
    }
    for (int c = 0; c < chunks; ++c) {
        const int k0 = c * kc, k1 = std::min(k0 + kc, a.k);
        MPI_Wait(&req[c], MPI_STATUS_IGNORE);  // chunk c has arrived
        if (my_rows == 0) continue;
        // C[my rows] += A[my rows, k0:k1] * B[k0:k1, :], in column blocks; between
        // blocks MPI_Test drives the progress of the next chunk's transfer
        // (Open MPI progresses non-blocking collectives only inside MPI calls)
        for (int j0 = 0; j0 < a.p; j0 += GEMM_BJ) {
            const int j1 = std::min(j0 + GEMM_BJ, a.p);
            gemm_add(my_rows, k0, k1, j0, j1, m1.data_ptr(), a.k, m2.data_ptr(), a.p,
                     m3.data_ptr(), a.p);
            if (c + 1 < chunks) {
                int done = 0;
                MPI_Test(&req[c + 1], &done, MPI_STATUS_IGNORE);
            }
        }
    }

    // collect matrix3 blocks on rank 0 (one collective call)
    std::vector<int> cnt(size), dsp(size);
    for (int r = 0; r < size; ++r) {
        cnt[r] = rows[r] * a.p;
        dsp[r] = offs[r] * a.p;
    }
    if (rank == 0)
        MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, m3.data_ptr(), cnt.data(), dsp.data(), MPI_INT, 0,
                    MPI_COMM_WORLD);
    else
        MPI_Gatherv(m3.data_ptr(), cnt[rank], MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0,
                    MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    t.mul = MPI_Wtime() - t0;

    // --- collect matrix1 on rank 0 for printing / verification (not timed) ---
    if (a.print || a.verify) {
        for (int r = 0; r < size; ++r) {
            cnt[r] = rows[r] * a.k;
            dsp[r] = offs[r] * a.k;
        }
        if (rank == 0)
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT, m1.data_ptr(), cnt.data(), dsp.data(), MPI_INT,
                        0, MPI_COMM_WORLD);
        else
            MPI_Gatherv(m1.data_ptr(), cnt[rank], MPI_INT, nullptr, nullptr, nullptr, MPI_INT, 0,
                        MPI_COMM_WORLD);
    }

    int rc = 0;
    if (rank == 0) {
        report(a, "mpi", "opt", size, t);
        rc = finish(a, m1, m2, m3);
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
