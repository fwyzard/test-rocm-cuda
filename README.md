# Test application using both CUDA and HIP/ROCm

### Clone the repository
```bash
git clone git@github.com:fwyzard/test-rocm-cuda.git
```

### Build the application
```bash
cd test-rocm-cuda
make
```

### Running using both backends
```bash
./test --rocm --cuda
```

On a machine without any NVIDIA or AMD GPU, the expected output is
```
Failed to initialise the ROCm runtime
Failed to initialise the CUDA runtime
```

On a machine with an NVIDIA GPU and the CUDA runtime, the expected output is
```
Failed to initialise the ROCm runtime
CUDA kernel ran successfully
```

On a machine with an AMD GPU and the HIP/ROCm runtime, the expected output is
```
ROCm kernel ran successfully
Failed to initialise the CUDA runtime
```

On a machine with both an NVIDIA and an AMD GPU and the corresponding runtimes, the expected output is
```
ROCm kernel ran successfully
CUDA kernel ran successfully
```
