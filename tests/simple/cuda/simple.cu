#define GROUPSIZE 64
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

__global__ void k(int *dobarrier) {
  int i = threadIdx.x;
  if (dobarrier[i]) {
    asm("bar.sync 0;"); // use asm to foil nvcc
  }
}

int main() {
  int *dobarrier = new int[GROUPSIZE];
  for (int i=0; i<GROUPSIZE; i++) {
    dobarrier[i] = (i % 2 == 0) ? 1 : 0;
  }

  int *d_dobarrier;
  size_t d_dobarrier_size = sizeof(int)*GROUPSIZE;
  cudaMalloc((void **)&d_dobarrier, d_dobarrier_size);
  cudaMemcpy(d_dobarrier, dobarrier, d_dobarrier_size, cudaMemcpyHostToDevice);


  ASSERT_NO_CUDA_ERROR();
  k<<<1, GROUPSIZE>>>(d_dobarrier);
  ASSERT_NO_CUDA_ERROR();

  cudaFree(d_dobarrier);
  delete[] dobarrier;
  return 0;
}

