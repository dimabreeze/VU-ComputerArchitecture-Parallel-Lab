// OpenCL (GPU) variant:
//   matrix1 fill    -> parallel on GPU (kernel fill_matrix1)
//   matrix2 fill    -> sequential on CPU, then copied to GPU
//   multiplication  -> parallel on GPU (kernel matmul)
// Kernel file: --kernel <file.cl> (default: src/opencl/base/naive.cl).
// Timings include host<->device transfers, but not kernel compilation.
// Every run is checked (not timed): the status of each kernel launch and a spot check
// of matrix3 on the CPU. A kernel killed on the GPU (seen at n = 14000: a "0.2 s" run
// of a 168 s job) now exits with an error instead of reporting a fake time.
#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "common.hpp"

#ifndef CL_KERNEL_DIR
#define CL_KERNEL_DIR "src/opencl"
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

// A finished command with a negative status failed on the device (e.g. was killed)
static void check_event(cl_event e, const char* what) {
    cl_int status = 0;
    CL_CHECK(clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status,
                            nullptr));
    clReleaseEvent(e);
    if (status != CL_COMPLETE) {
        std::fprintf(stderr, "GPU command failed: %s (status %d)\n", what, status);
        std::exit(1);
    }
}

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

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    if (a.kernel.empty()) a.kernel = CL_KERNEL_DIR "/base/naive.cl";

    // --- device setup (GPU) ---
    cl_platform_id platform;
    cl_device_id device;
    CL_CHECK(clGetPlatformIDs(1, &platform, nullptr));
    CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    char dev_name[256] = {0};
    cl_uint compute_units = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CL_CHECK(err);
    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
    CL_CHECK(err);

    // --- build kernels ---
    const std::string src = read_file(a.kernel);
    const char* src_ptr = src.c_str();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, nullptr, &err);
    CL_CHECK(err);
    if (clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        size_t len = 0;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string log(len, '\0');
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
        std::fprintf(stderr, "Kernel build failed:\n%s\n", log.c_str());
        return 1;
    }
    cl_kernel k_fill = clCreateKernel(prog, "fill_matrix1", &err);
    CL_CHECK(err);
    cl_kernel k_mul = clCreateKernel(prog, "matmul", &err);
    CL_CHECK(err);

    Matrix m1(a.n, a.k), m2(a.k, a.p), m3(a.n, a.p);
    const size_t bytes1 = m1.data.size() * sizeof(int);
    const size_t bytes2 = m2.data.size() * sizeof(int);
    const size_t bytes3 = m3.data.size() * sizeof(int);
    cl_mem d1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, bytes1, nullptr, &err);
    CL_CHECK(err);
    cl_mem d2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes2, nullptr, &err);
    CL_CHECK(err);
    cl_mem d3 = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes3, nullptr, &err);
    CL_CHECK(err);
    Times t;

    // --- matrix1 on GPU (parallel), copied back to host ---
    double t0 = now_sec();
    cl_ulong seed = a.seed;
    CL_CHECK(clSetKernelArg(k_fill, 0, sizeof(cl_mem), &d1));
    CL_CHECK(clSetKernelArg(k_fill, 1, sizeof(int), &a.n));
    CL_CHECK(clSetKernelArg(k_fill, 2, sizeof(int), &a.k));
    CL_CHECK(clSetKernelArg(k_fill, 3, sizeof(int), &a.x));
    CL_CHECK(clSetKernelArg(k_fill, 4, sizeof(cl_ulong), &seed));
    const size_t g_fill[2] = {static_cast<size_t>(a.n), static_cast<size_t>(a.k)};
    cl_event e_fill;
    CL_CHECK(clEnqueueNDRangeKernel(q, k_fill, 2, nullptr, g_fill, nullptr, 0, nullptr, &e_fill));
    CL_CHECK(clEnqueueReadBuffer(q, d1, CL_TRUE, 0, bytes1, m1.data.data(), 0, nullptr, nullptr));
    t.fill1 = now_sec() - t0;

    // --- matrix2 on CPU (sequential) ---
    t0 = now_sec();
    fill_matrix2_sequential(m2, a.seed);
    t.fill2 = now_sec() - t0;

    // --- multiplication on GPU: upload matrix2, run, download matrix3 ---
    t0 = now_sec();
    CL_CHECK(clEnqueueWriteBuffer(q, d2, CL_FALSE, 0, bytes2, m2.data.data(), 0, nullptr, nullptr));
    CL_CHECK(clSetKernelArg(k_mul, 0, sizeof(cl_mem), &d1));
    CL_CHECK(clSetKernelArg(k_mul, 1, sizeof(cl_mem), &d2));
    CL_CHECK(clSetKernelArg(k_mul, 2, sizeof(cl_mem), &d3));
    CL_CHECK(clSetKernelArg(k_mul, 3, sizeof(int), &a.n));
    CL_CHECK(clSetKernelArg(k_mul, 4, sizeof(int), &a.k));
    CL_CHECK(clSetKernelArg(k_mul, 5, sizeof(int), &a.p));
    const size_t g_mul[2] = {static_cast<size_t>(a.n), static_cast<size_t>(a.p)};
    cl_event e_mul;
    CL_CHECK(clEnqueueNDRangeKernel(q, k_mul, 2, nullptr, g_mul, nullptr, 0, nullptr, &e_mul));
    CL_CHECK(clEnqueueReadBuffer(q, d3, CL_TRUE, 0, bytes3, m3.data.data(), 0, nullptr, nullptr));
    t.mul = now_sec() - t0;

    // --- checks (not timed): kernel status (the blocking reads above succeed even if a
    // kernel failed) and a spot check of the result ---
    check_event(e_fill, "fill_matrix1");
    check_event(e_mul, "matmul");

    clReleaseMemObject(d1);
    clReleaseMemObject(d2);
    clReleaseMemObject(d3);
    clReleaseKernel(k_fill);
    clReleaseKernel(k_mul);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    if (!spot_check(a, m1, m2, m3)) return 3;

    std::string variant = a.kernel.substr(a.kernel.find_last_of('/') + 1);
    variant = variant.substr(0, variant.find('.'));
    report(a, "opencl", variant.c_str(), static_cast<int>(compute_units), t,
           std::string("compute units, device: ") + dev_name);
    return finish(a, m1, m2, m3);
}
