__kernel void k(__global int *dobarrier) {
  int i = get_local_id(0);
  if (dobarrier[i]) {
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}
