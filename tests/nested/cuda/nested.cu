#include <cstdio>

#define ASSERT_NO_CUDA_ERROR() {                              \
  cudaThreadSynchronize();                                    \
  cudaError_t err = cudaGetLastError();                       \
  if (err != cudaSuccess) {                                   \
    printf("Cuda error (%s/%d) in file '%s' in line %i\n",    \
           cudaGetErrorString(err), err, __FILE__, __LINE__); \
    exit(1);                                                  \
  }                                                           \
} while(0);

#define TRACE() {                                             \
  trace[(lid*x*y*8)+((loop)*8)+0] = A[0][0];                  \
  trace[(lid*x*y*8)+((loop)*8)+1] = A[0][1];                  \
  trace[(lid*x*y*8)+((loop)*8)+2] = A[0][2];                  \
  trace[(lid*x*y*8)+((loop)*8)+3] = A[0][3];                  \
  trace[(lid*x*y*8)+((loop)*8)+4] = A[1][0];                  \
  trace[(lid*x*y*8)+((loop)*8)+5] = A[1][1];                  \
  trace[(lid*x*y*8)+((loop)*8)+6] = A[1][2];                  \
  trace[(lid*x*y*8)+((loop)*8)+7] = A[1][3];                  \
  loop++;                                                     \
} while(0);

//set x and y through xyvals
__global__ void k2(int *xyvals, int *trace, int*final) {
  __shared__ int A[2][4];
  int buf, x, y, i, j;
  int lid = threadIdx.x;

  //initialize A
  if (lid == 0) {
    A[0][0] =  0; A[0][1] =  1; A[0][2] =  2; A[0][3] =  3;
    A[1][0] = -1; A[1][1] = -1; A[1][2] = -1; A[1][3] = -1;
  }
  __syncthreads();

  x = (lid == 0 ? xyvals[0] : xyvals[1]);
  y = (lid == 0 ? xyvals[1] : xyvals[0]);
  buf = i = 0;
  int loop = 0;
  while (i < x) {
    j = 0;
    while (j < y) {
      __syncthreads();
      TRACE();
      A[1-buf][lid] = A[buf][(lid+1)%4];
      buf = 1 - buf;
      j++;
    }
    i++;
  }

  __syncthreads();
  if (lid == 0) {
    final[0] = A[0][0]; final[1] = A[0][1];
    final[2] = A[0][2]; final[3] = A[0][3];
    final[4] = A[1][0]; final[5] = A[1][1];
    final[6] = A[1][2]; final[7] = A[1][3];
  }
}

#define GROUPSIZE 4

int main(int argc, char **argv) {

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
  int *d_xyvals;
  size_t d_xyvals_size = sizeof(int)*2;
  cudaMalloc((void **)&d_xyvals, d_xyvals_size);
  cudaMemcpy(d_xyvals, xyvals, d_xyvals_size, cudaMemcpyHostToDevice);

  // trace shared array A[] after each __syncthreads, for each thread
  // number of trace items :=
  //   8 values in A[]
  //   __syncthreads() hit (xyvals[0]*xyvals[1]) times
  //   by GROUPSIZE threads
  int ntrace = 8 * (xyvals[0]*xyvals[1]) * GROUPSIZE;
  int *trace = new int[ntrace];
  for (int i=0; i<ntrace; i++) {
    trace[i] = 99;
  }
  int *d_trace;
  size_t d_trace_size = sizeof(int)*ntrace;
  cudaMalloc((void **)&d_trace, d_trace_size);
  cudaMemcpy(d_trace, trace, d_trace_size, cudaMemcpyHostToDevice);

  // also record the final state of A
  int final[8];
  for (int i=0; i<8; i++) {
    final[i] = 99;
  }
  int *d_final;
  size_t d_final_size = sizeof(int)*8;
  cudaMalloc((void **)&d_final, d_final_size);
  cudaMemcpy(d_final, final, d_final_size, cudaMemcpyHostToDevice);

  // run kernel
  printf("Set x and y through xyvals[%d,%d]...", xyvals[0], xyvals[1]);
  ASSERT_NO_CUDA_ERROR();
  k2<<</*gridDim=*/1, GROUPSIZE>>>(d_xyvals, d_trace, d_final);
  ASSERT_NO_CUDA_ERROR();
  printf("[done]\n");

  // print out trace
  cudaMemcpy(trace, d_trace, d_trace_size, cudaMemcpyDeviceToHost);
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
  cudaMemcpy(final, d_final, d_final_size, cudaMemcpyDeviceToHost);
  printf("final state\n");
  printf("    A = {{%d,%d,%d,%d}, {%d,%d,%d,%d}}\n",
    final[0],final[1],final[2],final[3],
    final[4],final[5],final[6],final[7]);

  cudaFree(d_xyvals);
  cudaFree(d_trace);
  cudaFree(d_final);
  delete[] trace;
  return 0;

}

