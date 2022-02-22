#include <cassert>

#include <hip/hip_runtime.h>

namespace rocm {

  __global__ void executeKernel(int* __restrict__ d, const int N) {
    const auto i = threadIdx.x;
    if (i < N)
      atomicAdd(d, 1u);
  }

  bool enabled() {
    int count = 0;
    return (hipGetDeviceCount(&count) == hipSuccess and count > 0);
  }

  void execute() {
    int N = 10, *d = nullptr;
    const dim3 g{1}, l{1};
    assert(hipMalloc(&d, N * sizeof(unsigned int)) == hipSuccess);
    executeKernel<<<g, l, 0, 0>>>(d, N);
    assert(hipGetLastError() == hipSuccess);
    assert(hipFree(d) == hipSuccess);
  }

}  // namespace rocm
