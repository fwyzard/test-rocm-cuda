#include <cassert>
#include <iostream>

#include <hip/hip_runtime.h>

namespace rocm {

  __global__ void kernel(const int* __restrict__ data, int* __restrict__ result, const int size) {
    const auto i = threadIdx.x;
    if (i < size) {
      atomicAdd(result, data[i]);
    }
  }

  bool enabled() {
    int count = 0;
    return (hipGetDeviceCount(&count) == hipSuccess and count > 0);
  }

  void execute() {
    int size = 10;

    int device = 0;
    assert(hipSuccess == hipSetDevice(device));

    hipStream_t queue;
    assert(hipSuccess == hipStreamCreate(&queue));

    hipDeviceProp_t properties;
    assert(hipSuccess == hipGetDeviceProperties(&properties, device));
    std::cout << properties.name << std::endl;

    int* buffer_h;
    assert(hipSuccess == hipHostMalloc(&buffer_h, sizeof(int) * size, 0));
    for (int i = 0; i < size; ++i) {
      buffer_h[i] = 1;
    }
    int* result_h;
    assert(hipSuccess == hipHostMalloc(&result_h, sizeof(int), 0));
    *result_h = 0;

    int* buffer;
    assert(hipSuccess == hipMalloc(&buffer, sizeof(int) * size));
    assert(hipSuccess == hipMemcpyAsync(buffer, buffer_h, sizeof(int) * size, hipMemcpyDefault, queue));
    int* result;
    assert(hipSuccess == hipMalloc(&result, sizeof(int)));
    assert(hipSuccess == hipMemsetAsync(result, 0, sizeof(int), queue));

    kernel<<<1, 32, 0, queue>>>(buffer, result, size);
    assert(hipSuccess == hipGetLastError());
    assert(hipSuccess == hipMemcpyAsync(result_h, result, sizeof(int), hipMemcpyDefault, queue));
    assert(hipSuccess == hipStreamSynchronize(queue));

    std::cout << "result: " << *result_h << std::endl;

    assert(hipSuccess == hipStreamDestroy(queue));
    assert(hipSuccess == hipFree(result));
    assert(hipSuccess == hipFree(buffer));
    assert(hipSuccess == hipHostFree(result_h));
    assert(hipSuccess == hipHostFree(buffer_h));
  }

}  // namespace rocm
