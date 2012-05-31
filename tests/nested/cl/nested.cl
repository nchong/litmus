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

__kernel void k(__global int *xyvals, __global int *trace, __global int *final) {
  __local int A[2][4];
  int buf, x, y, i, j;
  int lid = get_local_id(0);

  //initialize A
  if (lid == 0) {
    A[0][0] =  0; A[0][1] =  1; A[0][2] =  2; A[0][3] =  3;
    A[1][0] = -1; A[1][1] = -1; A[1][2] = -1; A[1][3] = -1;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  x = (lid == 0 ? xyvals[0] : xyvals[1]);
  y = (lid == 0 ? xyvals[1] : xyvals[0]);
  buf = i = 0;
  int loop = 0;
  while (i < x) {
    j = 0;
    while (j < y) {
      barrier(CLK_LOCAL_MEM_FENCE);
      TRACE();
      A[1-buf][lid] = A[buf][(lid+1)%4];
      buf = 1 - buf;
      j++;
    }
    i++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0) {
    final[0] = A[0][0]; final[1] = A[0][1];
    final[2] = A[0][2]; final[3] = A[0][3];
    final[4] = A[1][0]; final[5] = A[1][1];
    final[6] = A[1][2]; final[7] = A[1][3];
  }
}
