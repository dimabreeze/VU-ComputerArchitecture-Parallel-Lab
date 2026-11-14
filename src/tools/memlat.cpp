// memlat: memory hierarchy microbenchmarks of this machine (docs/memory.md).
//
//   build/tools/memlat lat   latency of a dependent load vs working-set size:
//                            pointer chasing over 128-B cache lines in random order
//   build/tools/memlat tlb   latency vs number of 16-KB pages touched, one cache line per
//                            page (page-stride chasing): the line footprint stays small
//                            (fits L2), so a jump means TLB misses, not cache misses
//   build/tools/memlat bw    sequential read bandwidth vs number of threads (1 GB buffer)
//
// Output: CSV on stdout, header "mode,bytes,pages,threads,value";
// value = ns per load (lat, tlb) or GB/s (bw). Each point is the best of REPS runs.
//
// Pointer chasing: every node holds the address of the next one, and the order is a
// single random cycle (Sattolo's algorithm), so each load depends on the previous one
// and the hardware prefetcher cannot guess the next address. The time per step is
// the full latency of the level the data comes from.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static constexpr size_t LINE = 128;    // hw.cachelinesize
static constexpr size_t PAGE = 16384;  // hw.pagesize
static constexpr int REPS = 3;

using Clock = std::chrono::steady_clock;

static void* alloc_pages(size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, PAGE, bytes) != 0) {
        std::perror("posix_memalign");
        std::exit(1);
    }
    std::memset(p, 0, bytes);  // fault in every page before measuring
    return p;
}

// single random cycle over n nodes (Sattolo): next[order[i]] = order[i + 1]
static std::vector<size_t> random_cycle(size_t n, uint64_t seed) {
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::mt19937_64 rng(seed);
    for (size_t i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> d(0, i - 1);
        std::swap(order[i], order[d(rng)]);
    }
    return order;
}

// chase `steps` pointers from `start`, return ns per load (best of REPS)
static double chase(void* start, size_t steps) {
    void* volatile sink;
    void* p = start;
    for (size_t i = 0; i < steps / 4 + 1000; ++i) p = *static_cast<void**>(p);  // warm-up
    double best = 1e30;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = Clock::now();
        for (size_t i = 0; i < steps; i += 8) {
            p = *static_cast<void**>(p); p = *static_cast<void**>(p);
            p = *static_cast<void**>(p); p = *static_cast<void**>(p);
            p = *static_cast<void**>(p); p = *static_cast<void**>(p);
            p = *static_cast<void**>(p); p = *static_cast<void**>(p);
        }
        double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        best = std::min(best, ns / steps);
    }
    sink = p;
    (void)sink;
    return best;
}

// link node addresses into one random cycle
static void link(const std::vector<char*>& node, uint64_t seed) {
    auto order = random_cycle(node.size(), seed);
    for (size_t i = 0; i < order.size(); ++i)
        *reinterpret_cast<void**>(node[order[i]]) = node[order[(i + 1) % order.size()]];
}

static size_t steps_for(size_t nodes) { return std::max<size_t>(8'000'000, 4 * nodes); }

static void mode_lat() {
    std::vector<size_t> sizes;
    for (size_t s = 8 << 10; s <= (size_t(1) << 30); s *= 2) {
        sizes.push_back(s);
        if (s * 3 / 2 <= (size_t(1) << 30)) sizes.push_back(s * 3 / 2);
    }
    for (size_t bytes : sizes) {
        char* buf = static_cast<char*>(alloc_pages(bytes));
        std::vector<char*> node(bytes / LINE);
        for (size_t i = 0; i < node.size(); ++i) node[i] = buf + i * LINE;
        link(node, 42);
        double ns = chase(node[0], steps_for(node.size()));
        std::printf("lat,%zu,%zu,1,%.3f\n", bytes, (bytes + PAGE - 1) / PAGE, ns);
        std::fflush(stdout);
        std::free(buf);
    }
}

static void mode_tlb() {
    // from 16 pages: on a cycle of only 8 loads the core runs ahead (0.3 ns, below L1 latency)
    const size_t pages_list[] = {16, 32, 64, 96, 128, 160, 192, 256, 384, 512, 768, 1024,
                                 1536, 2048, 2560, 2816, 3072, 3328, 3584, 4096, 6144, 8192,
                                 12288, 16384, 32768};
    for (size_t pages : pages_list) {
        char* buf = static_cast<char*>(alloc_pages(pages * PAGE));
        std::vector<char*> node(pages);
        // one line per page; the line inside the page rotates so that the lines spread
        // over all cache sets (the same offset in every page would collide in a few sets)
        for (size_t i = 0; i < pages; ++i) node[i] = buf + i * PAGE + ((i * 37) % (PAGE / LINE)) * LINE;
        link(node, 7);
        double ns = chase(node[0], steps_for(pages));
        std::printf("tlb,%zu,%zu,1,%.3f\n", pages * PAGE, pages, ns);
        std::fflush(stdout);
        std::free(buf);
    }
}

static void mode_bw() {
    const size_t bytes = size_t(1) << 30;
    const size_t n = bytes / sizeof(uint64_t);
    auto* a = static_cast<uint64_t*>(alloc_pages(bytes));
    for (size_t i = 0; i < n; ++i) a[i] = i;
    const unsigned max_threads = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned t : {1u, 2u, 4u, 8u, 12u}) {
        if (t > max_threads) break;
        std::vector<uint64_t> part(t);
        double best = 0;
        for (int r = 0; r < REPS + 2; ++r) {
            std::vector<std::thread> th;
            auto t0 = Clock::now();
            for (unsigned k = 0; k < t; ++k)
                th.emplace_back([&, k] {
                    size_t lo = n * k / t, hi = n * (k + 1) / t;
                    uint64_t s = 0;
                    for (size_t i = lo; i < hi; ++i) s += a[i];
                    part[k] = s;
                });
            for (auto& x : th) x.join();
            double sec = std::chrono::duration<double>(Clock::now() - t0).count();
            best = std::max(best, bytes / sec / 1e9);
        }
        uint64_t check = 0;
        for (auto v : part) check += v;
        if (check != n * (n - 1) / 2) {
            std::fprintf(stderr, "bw: wrong sum\n");
            std::exit(2);
        }
        std::printf("bw,%zu,%zu,%u,%.2f\n", bytes, bytes / PAGE, t, best);
        std::fflush(stdout);
    }
    std::free(a);
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode != "lat" && mode != "tlb" && mode != "bw") {
        std::fprintf(stderr, "usage: %s lat|tlb|bw\n", argv[0]);
        return 1;
    }
    std::printf("mode,bytes,pages,threads,value\n");
    if (mode == "lat") mode_lat();
    if (mode == "tlb") mode_tlb();
    if (mode == "bw") mode_bw();
    return 0;
}
