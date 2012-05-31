#include "clwrapper.h"
#include <cstdio>
#include <iostream>

#define PLATFORM 0
#define DEVICE 0
#define GROUPSIZE 4

int main(int argc, char **argv) {
  // platform/device info
  std::cout << clinfo() << std::endl;

  // thread0 runs outer xyvals[0] times
  //              inner xyvals[1] times
  // other threads do opposite
  int xyvals[2];
  if (argc == 3) {
    xyvals[0] = atoi(argv[1]);
    xyvals[1] = atoi(argv[2]);
  } else {
    xyvals[0] = 4;
    xyvals[1] = 1;
  }

  // trace shared array A[] after each barrier, for each thread
  // number of trace items :=
  //   8 values in A[]
  //   __syncthreads() hit (xyvals[0]*xyvals[1]) times
  //   by GROUPSIZE threads
  int ntrace = 8 * (xyvals[0]*xyvals[1]) * GROUPSIZE;
  int *trace = new int[ntrace];
  for (int i=0; i<ntrace; i++) {
    trace[i] = 99;
  }

  // also record the final state of A
  int final[8];
  for (int i=0; i<8; i++) {
    final[i] = 99;
  }

  // creates a context and command queue
  CLWrapper clw(PLATFORM, DEVICE, /*profiling=*/false);

  // compile the OpenCL code
  const char *filename = "nested.cl";
  cl_program program = clw.compile(filename);

  // generate all kernels
  clw.create_all_kernels(program);

  // get handlers to kernels
  cl_kernel k = clw.kernel_of_name("k");

  // create some memory objects on the device
  cl_mem d_xyvals = clw.dev_malloc(sizeof(int)*2,      CL_MEM_READ_ONLY);
  cl_mem d_trace  = clw.dev_malloc(sizeof(int)*ntrace, CL_MEM_READ_WRITE);
  cl_mem d_final  = clw.dev_malloc(sizeof(int)*8,      CL_MEM_READ_WRITE);

  // memcpy into these objects
  clw.memcpy_to_dev(d_xyvals, sizeof(int)*2,      xyvals);
  clw.memcpy_to_dev(d_trace,  sizeof(int)*ntrace, trace);
  clw.memcpy_to_dev(d_final,  sizeof(int)*8,      final);

  // set kernel arguments
  clw.kernel_arg(k, d_xyvals, d_trace, d_final);

  // run the kernel
  cl_uint dim = 1;
  size_t global_work_size = GROUPSIZE;
  size_t local_work_size  = GROUPSIZE;
  clw.run_kernel(k, dim, &global_work_size, &local_work_size);

  // memcpy back trace
  clw.memcpy_from_dev(d_trace, sizeof(int)*ntrace, trace);

  // printout trace
  int stride = 8 * (xyvals[0]*xyvals[1]);
  for (int lid=0; lid<GROUPSIZE; lid++) {
    printf("lid = %d\n", lid);
    for (int xy=0; xy<(xyvals[0]*xyvals[1]); xy++) {
      printf("(%d) A = {{%d,%d,%d,%d}, {%d,%d,%d,%d}}\n",
        xy,
        trace[(lid*stride)+(xy*8)+0], trace[(lid*stride)+(xy*8)+1],
        trace[(lid*stride)+(xy*8)+2], trace[(lid*stride)+(xy*8)+3],
        trace[(lid*stride)+(xy*8)+4], trace[(lid*stride)+(xy*8)+5],
        trace[(lid*stride)+(xy*8)+6], trace[(lid*stride)+(xy*8)+7]
      );
    }
    printf("---\n");
  }

  // print out final state
  clw.memcpy_from_dev(d_final, sizeof(int)*8, final);
  printf("final state\n");
  printf("    A = {{%d,%d,%d,%d}, {%d,%d,%d,%d}}\n",
    final[0],final[1],final[2],final[3],
    final[4],final[5],final[6],final[7]);

  // clean up
  delete[] trace;
  return 0;
}

