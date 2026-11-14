// OpenCL (GPU) optimized host program; kernels in src/opencl/opt/*.cl
// (src/opencl/common.cl is prepended to the kernel file). Task semantics as in base:
//   matrix1 fill    -> parallel on GPU (kernel fill_matrix1)
//   matrix2 fill    -> sequential on CPU
//   multiplication  -> parallel on GPU (kernel matmul)
// Differences from base.cpp:
//   - zero-copy buffers (CL_MEM_USE_HOST_PTR + map/unmap): on Apple Silicon CPU and
//     GPU share memory, so matrices are not copied between host and device
//   - kernel geometry (work-group size, block per group) is reported by the kernel
//     file itself (kernel `geometry`), so one host serves all kernels
//   - the multiplication is launched in bands of rows: every launch is short, which
//     keeps macOS from killing a long-running GPU command (see docs/opencl_opt.md)
//   - every launch is checked: event status after completion + spot check of
//     matrix3 on the CPU; a failed run exits with an error instead of reporting a
//     fake time (base.cpp could silently report 0.2 s for a 168 s job)
//   - profiling: pure kernel time is shown in the report (not in the CSV)
//   - mad24 integer multiply-add when the values allow it (-DUSE_MAD24)
// Kernel tuning without editing files: CL_BUILD_OPTS="-DTS=32" build/opencl-opt ...
// Analysis and measurements: docs/opencl_opt.md
#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <mach/mach_time.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"

#ifndef CL_KERNEL_DIR
#define CL_KERNEL_DIR "src/opencl"
#endif
#ifndef BAND_OPS
#define BAND_OPS (1ull << 34)  // max multiply-adds per kernel launch (~1.7e10)
#endif

#define CL_CHECK(call)                                                              \
    do {                                                                            \
        cl_int err_ = (call);                                                       \
        if (err_ != CL_SUCCESS) {                                                   \
            std::fprintf(stderr, "OpenCL error %d at %s:%d: %s\n", err_, __FILE__, \
                         __LINE__, #call);                                          \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "Cannot open kernel file: %s\n", path.c_str());
        std::exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Seconds between start and end of a finished command (profiling queue).
// Apple's OpenCL returns profiling timestamps in mach_absolute_time ticks, not in
// nanoseconds as the standard says (1 tick = 125/3 ns on Apple Silicon), so the
// difference is converted with mach_timebase_info.
static double event_seconds(cl_event e) {
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
    double ns = static_cast<double>(t1 - t0);
#ifdef __APPLE__
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    ns = ns * tb.numer / tb.denom;
#endif
    return ns * 1e-9;
}

// A finished command with a negative status failed on the device (e.g. was killed)
static void check_event(cl_event e, const char* what) {
    cl_int status = 0;
    CL_CHECK(clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status,
                            nullptr));
    if (status != CL_COMPLETE) {
        std::fprintf(stderr, "GPU command failed: %s (status %d)\n", what, status);
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.kernel.empty()) a.kernel = CL_KERNEL_DIR "/opt/regblock.cl";

    // --- device setup (GPU) ---
    cl_platform_id platform;
    cl_device_id device;
    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    char dev_name[256] = {0};
    cl_uint compute_units = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units,
                    nullptr);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CL_CHECK(err);

    // --- build: common.cl + kernel file ---
    const std::string common_src = read_file(CL_KERNEL_DIR "/common.cl");
    const std::string kernel_src = read_file(a.kernel);
    const char* srcs[2] = {common_src.c_str(), kernel_src.c_str()};
    cl_program prog = clCreateProgramWithSource(ctx, 2, srcs, nullptr, &err);
    CL_CHECK(err);
    std::string opts;
    if (a.x < (1 << 23)) opts += "-DUSE_MAD24 ";  // m1 in [1, x], m2 in {-1, 1}: 24-bit safe
    if (const char* extra = std::getenv("CL_BUILD_OPTS")) opts += extra;
    if (clBuildProgram(prog, 1, &device, opts.c_str(), nullptr, nullptr) != CL_SUCCESS) {
        size_t len = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::fprintf(stderr, "Kernel build failed (%s):\n%s\n", opts.c_str(), log.c_str());
        return 1;
    }
    cl_kernel k_fill = clCreateKernel(prog, "fill_matrix1", &err);
    CL_CHECK(err);
    cl_kernel k_mul = clCreateKernel(prog, "matmul", &err);
    CL_CHECK(err);
    cl_kernel k_geo = clCreateKernel(prog, "geometry", &err);
    CL_CHECK(err);

    // --- kernel geometry: {LX, LY, TM, TN} ---
    int geo[4] = {0, 0, 0, 0};
    {
        cl_mem dg = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(geo), nullptr, &err);
        CL_CHECK(err);
        CL_CHECK(clSetKernelArg(k_geo, 0, sizeof(cl_mem), &dg));
        const size_t one = 1;
        CL_CHECK(clEnqueueNDRangeKernel(q, k_geo, 1, nullptr, &one, nullptr, 0, nullptr, nullptr));
        CL_CHECK(clEnqueueReadBuffer(q, dg, CL_TRUE, 0, sizeof(geo), geo, 0, nullptr, nullptr));
        clReleaseMemObject(dg);
    }
    const int LX = geo[0], LY = geo[1], TM = geo[2], TN = geo[3];
    size_t max_wg = 0;
    clGetKernelWorkGroupInfo(k_mul, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg,
                             nullptr);
    if (static_cast<size_t>(LX) * LY > max_wg) {
        std::fprintf(stderr, "Work-group %dx%d exceeds the kernel limit %zu\n", LX, LY, max_wg);
        return 1;
    }

    // --- matrices (host memory) and zero-copy device buffers on top of them ---
    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    const size_t bytes1 = m1.data.size() * sizeof(int);
    const size_t bytes2 = m2.data.size() * sizeof(int);
    const size_t bytes3 = m3.data.size() * sizeof(int);
    Times t;

    // --- matrix1 on GPU (parallel), written straight into host memory ---
    double t0 = now_sec();
    cl_mem d1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes1,
                               m1.data_ptr(), &err);
    CL_CHECK(err);
    cl_ulong seed = a.seed;
    CL_CHECK(clSetKernelArg(k_fill, 0, sizeof(cl_mem), &d1));
    CL_CHECK(clSetKernelArg(k_fill, 1, sizeof(int), &a.n));
    CL_CHECK(clSetKernelArg(k_fill, 2, sizeof(int), &a.k));
    CL_CHECK(clSetKernelArg(k_fill, 3, sizeof(int), &a.x));
    CL_CHECK(clSetKernelArg(k_fill, 4, sizeof(cl_ulong), &seed));
    const size_t g_fill[2] = {static_cast<size_t>(a.k), static_cast<size_t>(a.n)};
    cl_event e_fill;
    CL_CHECK(clEnqueueNDRangeKernel(q, k_fill, 2, nullptr, g_fill, nullptr, 0, nullptr, &e_fill));
    // map = make the GPU result visible in m1 on the host (no copy with zero-copy buffers)
    void* p1 = clEnqueueMapBuffer(q, d1, CL_TRUE, CL_MAP_READ, 0, bytes1, 0, nullptr, nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(clEnqueueUnmapMemObject(q, d1, p1, 0, nullptr, nullptr));
    CL_CHECK(clFinish(q));
    check_event(e_fill, "fill_matrix1");
    t.fill1 = now_sec() - t0;

    // --- matrix2 on CPU (sequential) ---
    t0 = now_sec();
    fill_matrix2_sequential(m2, a.seed);
    t.fill2 = now_sec() - t0;

    // --- multiplication on GPU, in bands of rows ---
    t0 = now_sec();
    cl_mem d2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bytes2,
                               m2.data_ptr(), &err);
    CL_CHECK(err);
    cl_mem d3 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, bytes3,
                               m3.data_ptr(), &err);
    CL_CHECK(err);
    CL_CHECK(clSetKernelArg(k_mul, 0, sizeof(cl_mem), &d1));
    CL_CHECK(clSetKernelArg(k_mul, 1, sizeof(cl_mem), &d2));
    CL_CHECK(clSetKernelArg(k_mul, 2, sizeof(cl_mem), &d3));
    CL_CHECK(clSetKernelArg(k_mul, 3, sizeof(int), &a.n));
    CL_CHECK(clSetKernelArg(k_mul, 4, sizeof(int), &a.k));
    CL_CHECK(clSetKernelArg(k_mul, 5, sizeof(int), &a.p));

    // rows per launch: a multiple of TM, at most BAND_OPS multiply-adds per launch
    const unsigned long long per_row = static_cast<unsigned long long>(a.k) * a.p;
    int band = static_cast<int>(std::min<unsigned long long>(a.n, BAND_OPS / per_row));
    band = std::max(TM, band / TM * TM);
    const size_t lws[2] = {static_cast<size_t>(LX), static_cast<size_t>(LY)};
    const size_t gx = static_cast<size_t>((a.p + TN - 1) / TN) * LX;
    std::vector<cl_event> e_mul;
    for (int row0 = 0; row0 < a.n; row0 += band) {
        const int rows = std::min(band, a.n - row0);
        const size_t gws[2] = {gx, static_cast<size_t>((rows + TM - 1) / TM) * LY};
        CL_CHECK(clSetKernelArg(k_mul, 6, sizeof(int), &row0));
        cl_event e;
        CL_CHECK(clEnqueueNDRangeKernel(q, k_mul, 2, nullptr, gws, lws, 0, nullptr, &e));
        CL_CHECK(clFlush(q));  // submit now, so bands run back to back
        e_mul.push_back(e);
    }
    void* p3 = clEnqueueMapBuffer(q, d3, CL_TRUE, CL_MAP_READ, 0, bytes3, 0, nullptr, nullptr, &err);
    CL_CHECK(err);
    CL_CHECK(clEnqueueUnmapMemObject(q, d3, p3, 0, nullptr, nullptr));
    CL_CHECK(clFinish(q));
    t.mul = now_sec() - t0;

    double kernel_time = 0;
    for (cl_event e : e_mul) {
        check_event(e, "matmul");
        kernel_time += event_seconds(e);
        clReleaseEvent(e);
    }
    clReleaseEvent(e_fill);

    clReleaseMemObject(d1);
    clReleaseMemObject(d2);
    clReleaseMemObject(d3);
    clReleaseKernel(k_fill);
    clReleaseKernel(k_mul);
    clReleaseKernel(k_geo);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    // --- cheap correctness check on every run (not timed) ---
    if (!spot_check(a, m1, m2, m3)) return 3;

    std::string variant = a.kernel.substr(a.kernel.find_last_of('/') + 1);
    variant = variant.substr(0, variant.find('.'));
    char note[512];
    std::snprintf(note, sizeof(note),
                  "compute units, device: %s\n  kernel %s, work-group %dx%d, block %dx%d, "
                  "%zu launch(es), pure kernel time %.6f s, build options: %s",
                  dev_name, variant.c_str(), LX, LY, TM, TN, e_mul.size(), kernel_time,
                  opts.c_str());
    report(a, "opencl", variant.c_str(), static_cast<int>(compute_units), t, note);
    return finish(a, m1, m2, m3);
}
