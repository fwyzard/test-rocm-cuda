.PHONY: all clean

COMMA=,

CXX := g++
CXX_FLAGS := -std=c++17 -O2 -g
CXX_HOST_FLAGS := -Wall -fPIC

HIP_BASE := /cvmfs/patatrack.cern.ch/externals/x86_64/rhel8/amd/rocm-5.0.2
HIPCC := $(HIP_BASE)/hip/bin/hipcc --rocm-path=$(HIP_BASE)
HIP_CXXFLAGS := -I$(HIP_BASE)/include
HIP_LDFLAGS := -L$(HIP_BASE)/lib -L$(HIP_BASE)/lib64 -lamdhip64 -lamd_comgr -lhsa-runtime64 -Wl,-rpath,$(HIP_BASE)/lib -Wl,-rpath,$(HIP_BASE)/lib64

CUDA_BASE := /cvmfs/patatrack.cern.ch/externals/x86_64/rhel8/nvidia/cuda-11.5.2
CUDA_DRIVER := /cvmfs/patatrack.cern.ch/externals/x86_64/rhel8/nvidia/compat/510.47.03/
NVCC := $(CUDA_BASE)/bin/nvcc -ccbin ${CXX}
CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
CUDA_LDFLAGS := -L$(CUDA_BASE)/lib64 -L$(CUDA_DRIVER) -lcudart -lcuda
CUDA_HOST_LDFLAGS := -Wl,-rpath,$(CUDA_BASE)/lib64 -Wl,-rpath,$(CUDA_DRIVER)


all: test test.static

clean:
	rm -f test test.static *.o *.so


backend-rocm.o: backend-rocm.cc
	$(HIPCC) $(CXX_FLAGS) $(CXX_HOST_FLAGS) $(HIP_CXXFLAGS) $< -c -o $@

libbackend-rocm.so: backend-rocm.o
	$(HIPCC) $(CXX_FLAGS) $(CXX_HOST_FLAGS) $< -shared $(HIP_LDFLAGS) -o $@

backend-cuda.o: backend-cuda.cc
	$(NVCC) $(CXX_FLAGS) -Xcompiler "$(CXX_HOST_FLAGS)" $(CUDA_CXXFLAGS) $< -x cu -c -o $@

libbackend-cuda.so: backend-cuda.o
	$(NVCC) $(CXX_FLAGS) $(CUDA_LDFLAGS) -Xcompiler "$(CXX_HOST_FLAGS) $(subst $(COMMA),\$(COMMA),$(CUDA_HOST_LDFLAGS))" $< -shared -o $@

test.o: test.cc
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) -c test.cc -o test.o

test: test.o libbackend-rocm.so libbackend-cuda.so
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) -o test $< -L. -Wl,-rpath,. -lbackend-rocm -lbackend-cuda

test.static: test.o backend-rocm.o backend-cuda.o
	$(CXX) $(CXX_FLAGS) $(CXX_HOST_FLAGS) $^ $(CUDA_LDFLAGS) $(CUDA_HOST_LDFLAGS) $(HIP_LDFLAGS) -o $@
