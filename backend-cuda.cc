#include <cassert>

#include <cuda_runtime.h>

namespace cuda {

  __global__ void executeKernel(int* __restrict__ d, const int N) {
    const auto i = threadIdx.x;
    if (i < N)
      atomicAdd(d, 1u);
  }

  bool enabled() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess and count > 0);
  }

  void execute() {
    int N = 10, *d = nullptr;
    const dim3 g{1}, l{1};
    assert(cudaMalloc(&d, N * sizeof(unsigned int)) == cudaSuccess);
    executeKernel<<<g, l, 0, 0>>>(d, N);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaFree(d) == cudaSuccess);
  }

}  // namespace cuda
