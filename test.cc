#include <cstring>
#include <iostream>

#include "backend-cuda.h"
#include "backend-rocm.h"

int main(int argc, const char** argv) {
  for (int i = 1; i < argc; ++i) {
    // HIP with ROCm backend
    if (std::strcmp(argv[i], "--rocm") == 0) {
      if (rocm::enabled()) {
        rocm::execute();
        std::cout << "ROCm kernel ran successfully" << std::endl;
      } else {
        std::cout << "Failed to initialise the ROCm runtime" << std::endl;
      }
      continue;
    }
    // CUDA backend
    if (std::strcmp(argv[i], "--cuda") == 0) {
      if (cuda::enabled()) {
        cuda::execute();
        std::cout << "CUDA kernel ran successfully" << std::endl;
      } else {
        std::cout << "Failed to initialise the CUDA runtime" << std::endl;
      }
      continue;
    }
    // unknown option
    std::cout << argv[0] << ": unrecognised option " << argv[i] << std::endl;
  }

  return 0;
}
