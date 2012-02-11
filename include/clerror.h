#ifndef CL_ERROR_HANDLING_H
#define CL_ERROR_HANDLING_H

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#elif __linux__
  #include <CL/cl.h>
#else
  #error Not sure where to find OpenCL header
#endif

#include "log.h"
#include <cassert>

const char *clGetErrorString(cl_int err);
const char *clGetBuildStatusString(cl_build_status status);

#define ASSERT_NO_CL_ERROR( callReturningErrorstatus ) {          \
  cl_int err = callReturningErrorstatus;                         \
  if (err != CL_SUCCESS) {                                        \
    fprintf(                                                      \
      stderr,                                                     \
      "%s:%d: %s(%d)\n", __FILE__, __LINE__, clGetErrorString(err), err);  \
    exit(1);                                                      \
  }                                                               \
} while(0) ;

#endif
