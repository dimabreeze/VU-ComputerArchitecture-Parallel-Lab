// MPI base variant: the course template ("MPI program sample", docs/Reference/) made
// correct and adapted to this platform. The multiplication algorithm and the work
// distribution are those of the template:
//   multiplication  -> parallel: each process computes rows i = rank, rank+size, ...
//                      (cyclic split, i-j-z loop), rows are sent back to rank 0 with
//                      MPI_Send / MPI_Recv
// Fixed compared to the template:
//   - matrix generation follows the task statement and gives the same data on all ranks
//     (in the template every process generated its own rand() matrices):
//       matrix1 fill -> parallel: each process fills its own rows
//       matrix2 fill -> sequential on rank 0, then MPI_Bcast (so fill2 includes the Bcast)
//   - one message tag per stream (template tags 10+i overflow MPI_TAG_UB for
//     large n); MPI guarantees ordering between the same pair of processes
//   - removed the stray '}' before MPI_Finalize, only rank 0 prints the report
// Added for the platform: command line arguments (Args), deterministic generators
// (common.hpp), --verify, CSV report.
#include <mpi.h>

#include "common.hpp"

static constexpr int TAG_M1 = 1;
static constexpr int TAG_M3 = 3;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Args a = parse_args(argc, argv);
    // all ranks must use the same seed (time-based seed may differ between ranks)
    MPI_Bcast(&a.seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    Times t;

    // --- matrix1: each process fills its own rows (parallel) ---
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = rank; i < a.n; i += size)
        for (int j = 0; j < a.k; ++j)
            m1.at(i, j) = rand_1_to_x(a.seed, static_cast<uint64_t>(i) * a.k + j, a.x);
    MPI_Barrier(MPI_COMM_WORLD);
    t.fill1 = MPI_Wtime() - t0;

    // --- matrix2: sequential on rank 0, then broadcast ---
    t0 = MPI_Wtime();
    if (rank == 0) fill_matrix2_sequential(m2, a.seed);
    MPI_Bcast(m2.data.data(), a.k * a.p, MPI_INT, 0, MPI_COMM_WORLD);
    t.fill2 = MPI_Wtime() - t0;

    // --- multiplication: each process computes its rows, rank 0 collects ---
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int i = rank; i < a.n; i += size)
        for (int j = 0; j < a.p; ++j) {
            int sum = 0;
            for (int z = 0; z < a.k; ++z) sum += m1.at(i, z) * m2.at(z, j);
            m3.at(i, j) = sum;
        }

    if (rank != 0) {
        for (int i = rank; i < a.n; i += size)
            MPI_Send(m3.row(i), a.p, MPI_INT, 0, TAG_M3, MPI_COMM_WORLD);
    } else {
        for (int src = 1; src < size; ++src)
            for (int i = src; i < a.n; i += size)
                MPI_Recv(m3.row(i), a.p, MPI_INT, src, TAG_M3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t.mul = MPI_Wtime() - t0;

    // --- collect matrix1 rows on rank 0 for printing / verification (not timed) ---
    if (a.print || a.verify) {
        if (rank != 0) {
            for (int i = rank; i < a.n; i += size)
                MPI_Send(m1.row(i), a.k, MPI_INT, 0, TAG_M1, MPI_COMM_WORLD);
        } else {
            for (int src = 1; src < size; ++src)
                for (int i = src; i < a.n; i += size)
                    MPI_Recv(m1.row(i), a.k, MPI_INT, src, TAG_M1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    int rc = 0;
    if (rank == 0) {
        report(a, "mpi", "base", size, t);
        rc = finish(a, m1, m2, m3);
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
