#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

namespace cuda {

  __global__ void kernel(const int* __restrict__ data, int* __restrict__ result, const int size) {
    const auto i = threadIdx.x;
    if (i < size) {
      atomicAdd(result, data[i]);
    }
  }

  bool enabled() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess and count > 0);
  }

  void execute() {
    int size = 10;

    int device = 0;
    assert(cudaSuccess == cudaSetDevice(device));

    cudaStream_t queue;
    assert(cudaSuccess == cudaStreamCreate(&queue));

    cudaDeviceProp properties;
    assert(cudaSuccess == cudaGetDeviceProperties(&properties, device));
    std::cout << properties.name << std::endl;

    int* buffer_h;
    assert(cudaSuccess == cudaMallocHost(&buffer_h, sizeof(int) * size));
    for (int i = 0; i < size; ++i) {
      buffer_h[i] = 1;
    }
    int* result_h;
    assert(cudaSuccess == cudaMallocHost(&result_h, sizeof(int)));
    *result_h = 0;

    int* buffer;
    assert(cudaSuccess == cudaMalloc(&buffer, sizeof(int) * size));
    assert(cudaSuccess == cudaMemcpyAsync(buffer, buffer_h, sizeof(int) * size, cudaMemcpyDefault, queue));
    int* result;
    assert(cudaSuccess == cudaMalloc(&result, sizeof(int)));
    assert(cudaSuccess == cudaMemsetAsync(result, 0, sizeof(int), queue));

    kernel<<<1, 32, 0, queue>>>(buffer, result, size);
    assert(cudaSuccess == cudaGetLastError());
    assert(cudaSuccess == cudaMemcpyAsync(result_h, result, sizeof(int), cudaMemcpyDefault, queue));
    assert(cudaSuccess == cudaStreamSynchronize(queue));

    std::cout << "result: " << *result_h << std::endl;

    assert(cudaSuccess == cudaStreamDestroy(queue));
    assert(cudaSuccess == cudaFree(result));
    assert(cudaSuccess == cudaFree(buffer));
    assert(cudaSuccess == cudaFreeHost(result_h));
    assert(cudaSuccess == cudaFreeHost(buffer_h));
  }

}  // namespace cuda
