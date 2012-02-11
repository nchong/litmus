#include "clwrapper.h"
#include <cstdio>
#include <iostream>

#define PLATFORM 0
#define DEVICE 0
#define GROUPSIZE 64

int main() {
  // platform/device info
  std::cout << clinfo() << std::endl;

  int *dobarrier = new int[GROUPSIZE];
  for (int i=0; i<GROUPSIZE; i++) {
    dobarrier[i] = (i % 2 == 0) ? 1 : 0;
  }

  // creates a context and command queue
  CLWrapper clw(PLATFORM, DEVICE, /*profiling=*/false);

  // compile the OpenCL code
  const char *filename = "simple.cl";
  cl_program program = clw.compile(filename);

  // generate all kernels
  clw.create_all_kernels(program);

  // get handlers to kernels
  cl_kernel k = clw.kernel_of_name("k");

  // create some memory objects on the device
  size_t d_dobarrier_size = sizeof(int)*GROUPSIZE;
  cl_mem d_dobarrier = clw.dev_malloc(d_dobarrier_size, CL_MEM_READ_ONLY);

  // memcpy into these objects
  clw.memcpy_to_dev(d_dobarrier, d_dobarrier_size, dobarrier);

  // set kernel arguments
  clw.kernel_arg(k, d_dobarrier);

  // run the kernel
  cl_uint dim = 1;
  size_t global_work_size = GROUPSIZE;
  size_t local_work_size  = GROUPSIZE;
  clw.run_kernel(k, dim, &global_work_size, &local_work_size);

  // clean up
  delete[] dobarrier;
  return 0;
}
