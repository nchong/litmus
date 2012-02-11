#include "clwrapper.h"
#include "clerror.h"
#include "log.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

CLWrapper::CLWrapper(int p, int d, bool profiling, bool all_devices) :
  num_platforms(0), platforms(NULL), p(p),
  num_devices(0), devices(NULL), d(d),
  profiling(profiling) {
  LOG(LOG_INFO, "Initializing context and command queue for device %d on platform %d", d, p);

  num_platforms = query_num_platforms();
  assert(p < (int)num_platforms);
  platforms = get_platform_list();      

  num_devices = query_num_devices(platforms[p]);
  devices = get_device_list(platforms[p]);
  assert(d < (int)num_devices);

  attach_context(all_devices);
  attach_command_queue(profiling ? CL_QUEUE_PROFILING_ENABLE : 0);
}

CLWrapper::~CLWrapper() {
  free_all_memobjs();
  if (command_queue) {
    clReleaseCommandQueue(command_queue);
  }
  if (context) {
    clReleaseContext(context);
  }
  LOG(LOG_INFO, "Releasing %d program%s", (int)programs.size(), programs.size() == 1 ? "" : "s");
  for (int i=0; i<(int)programs.size(); i++) {
    ASSERT_NO_CL_ERROR(clReleaseProgram(programs.at(i)));
  }
  programs.clear();
  LOG(LOG_INFO, "Releasing %d kernel%s", (int)kernelmap.size(), kernelmap.size() == 1 ? "" : "s");
  map<string, cl_kernel>::iterator i;
  for (i=kernelmap.begin(); i != kernelmap.end(); i++) {
    ASSERT_NO_CL_ERROR(clReleaseKernel(i->second));
  }
  kernelmap.clear();
  delete[] devices;
}

void CLWrapper::flush_command_queue() {
  ASSERT_NO_CL_ERROR(clFinish(command_queue));
}

bool CLWrapper::has_profiling() {
  return profiling;
}

cl_program &CLWrapper::compile_from_string(const char *program_string,
    const string &extra_flags, bool all_devices) {
  cl_int ret;
  //lengths=NULL -> program_string is null terminated
  cl_program program = clCreateProgramWithSource(context, /*count=*/1, (const char **) &program_string, /*lengths=*/NULL, &ret );
  assert(program);
  ASSERT_NO_CL_ERROR(ret);
  programs.push_back(program);

  stringstream flags;
  flags << extra_flags;

  // Math intrinsics options
  //flags << " -cl-single-precision-constant";
  //flags << " -cl-denorms-are-zero";

  // Optimization options
  //flags << " -cl-opt-disable";
  //flags << " -cl-strict-aliasing";
  //flags << " -cl-mad-enable";
  //flags << " -cl-no-signed-zeros";
  //flags << " -cl-unsafe-math-optimizations";
  //flags << " -cl-finite-math-only";
  //flags << " -cl-fast-relaxed-math";

  // Warnings suppress/request
  //flags << " -w";
  flags << " -Werror";

  cl_uint ndev = (all_devices ? num_devices : 1);
  cl_device_id *dev = (all_devices ? devices : &devices[d]);
  //pfn_notify=NULL -> call is blocking
  cl_uint builderr = clBuildProgram(program, ndev, dev, flags.str().c_str(), /*pfn_notify=*/NULL, /*user_data=*/NULL);

  //print out build logs
  if (builderr != CL_SUCCESS) {
    printf("> builderr: %s\n", clGetErrorString(builderr));

    for (int i=0; i<(int)num_devices; i++) {
      if (all_devices || i == d) {
        printf("> ---------\n");
        printf("> device %d\n", i);
        printf("> ---------\n");
        cl_build_status status;
        ASSERT_NO_CL_ERROR(
          clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL));
        printf(">\t build_status: %s\n", clGetBuildStatusString(status));

        size_t size;
        ASSERT_NO_CL_ERROR(
          clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &size));
        char *build_log = new char[size+1];
        ASSERT_NO_CL_ERROR(
          clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, size, build_log, NULL));
        build_log[size] = '\0';
        printf(">\t build_log:\n");
        printf("%s\n", build_log);
        printf("-----\n");
        delete[] build_log;
      }
    }

    exit(1);
  }

  return programs.back();

}

cl_program &CLWrapper::compile(const char *fname,
    const string &extra_flags, bool all_devices) {
  if (all_devices) {
    LOG(LOG_INFO, "Compiling file <%s> for all devices", fname);
  } else {
    LOG(LOG_INFO, "Compiling file <%s> for device %d", fname, d);
  }

  // read in file to string (via buf)
  ifstream file(fname, ios::binary);
  if (!file.is_open()) {
    LOG(LOG_FATAL, "Unable to open file <%s>.", fname);
  }
  std::ostringstream buf;
  buf << file.rdbuf() << endl;
  string s = buf.str();

  return compile_from_string(s.c_str(), extra_flags, all_devices);
}

cl_mem &CLWrapper::dev_malloc(size_t size, cl_mem_flags flags) {
  cl_int ret;
  cl_mem m = clCreateBuffer(context, flags, size, /*host_ptr*/NULL, &ret);
  ASSERT_NO_CL_ERROR(ret);
  memobjs.push_back(m);
  return memobjs.back();
}

void CLWrapper::dev_free(cl_mem m) {
  vector<cl_mem>::iterator it = find(memobjs.begin(), memobjs.end(), m);
  if (it == memobjs.end()) {
    LOG(LOG_WARN, "Freeing memory object not found in [memobjs]");
  } else {
    memobjs.erase(it);
  }
  ASSERT_NO_CL_ERROR(clReleaseMemObject(m));
}

void CLWrapper::free_all_memobjs() {
  LOG(LOG_INFO, "Freeing %d memobject%s", (int)memobjs.size(), memobjs.size() == 1 ? "" : "s");
  for (int i=0; i<(int)memobjs.size(); i++) {
    dev_free(memobjs.at(i));
  }
}

float CLWrapper::memcpy_to_dev(cl_mem buffer, size_t size, const void *ptr, size_t offset) {
  cl_bool blocking_write = CL_TRUE;
  cl_uint num_events_in_wait_list = 0;
  cl_event *event_wait_list = NULL;
  cl_event e;
  ASSERT_NO_CL_ERROR(
    clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, size, ptr, num_events_in_wait_list, event_wait_list, &e));
  if (profiling) {
    return time_and_release_event(e);
  } else {
    return 0;
  }
}

float CLWrapper::memcpy_from_dev(cl_mem buffer, size_t size, void *ptr, size_t offset) {
  cl_bool blocking_read = CL_TRUE;
  cl_uint num_events_in_wait_list = 0;
  cl_event *event_wait_list = NULL;
  cl_event e;
  ASSERT_NO_CL_ERROR(
    clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, &e));
  if (profiling) {
    return time_and_release_event(e);
  } else {
    return 0;
  }
}

float CLWrapper::copy_buffer(cl_mem src, cl_mem dst, size_t cb) {
  size_t src_offset = 0;
  size_t dst_offset = 0;
  cl_uint num_events_in_wait_list = 0;
  const cl_event *event_wait_list = NULL;
  cl_event e;
  ASSERT_NO_CL_ERROR(
    clEnqueueCopyBuffer(command_queue, src, dst, src_offset, dst_offset, cb, num_events_in_wait_list, event_wait_list, &e));
  return profiling ? time_and_release_event(e) : 0;
}

void CLWrapper::create_all_kernels(cl_program program) {
  cl_uint num_kernels;
  ASSERT_NO_CL_ERROR(
    clCreateKernelsInProgram(program, /*num_kernels=*/0, /*kernels=*/NULL, &num_kernels));
  cl_kernel *kernels = new cl_kernel[num_kernels];
  assert(kernels);
  LOG(LOG_INFO, "Creating %d kernels", num_kernels);
  ASSERT_NO_CL_ERROR(
    clCreateKernelsInProgram(program, num_kernels, kernels, NULL));
  for (int i=0; i<(int)num_kernels; i++) {
    string kernel_name = name_of_kernel(kernels[i]);
    map<string,cl_kernel>::iterator it = kernelmap.find(kernel_name);
    if (it != kernelmap.end()) {
      LOG(LOG_ERR, "Kernel name clash [%s]", kernel_name.c_str());
    }
    kernelmap.insert(make_pair(kernel_name, kernels[i]));
  }
  delete[] kernels;
}

cl_kernel &CLWrapper::create_kernel(cl_program program, const char*kernel_name) {
  cl_int ret;
  cl_kernel k = clCreateKernel(program, kernel_name, &ret);
  ASSERT_NO_CL_ERROR(ret);
  map<string,cl_kernel>::iterator it = kernelmap.find(string(kernel_name));
  if (it != kernelmap.end()) {
    LOG(LOG_ERR, "Kernel name clash [%s]", kernel_name);
  }
  kernelmap.insert(make_pair(kernel_name, k));
  return kernel_of_name(kernel_name);
}

cl_kernel &CLWrapper::kernel_of_name(const string name) {
  map<string,cl_kernel>::iterator it = kernelmap.find(name);
  if (it == kernelmap.end()) {
    LOG(LOG_FATAL, "Could not find kernel [%s]", name.c_str());
  }
  return it->second;
}

const string CLWrapper::name_of_kernel(cl_kernel kernel) {
  size_t size;
  ASSERT_NO_CL_ERROR(
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, /*size=*/0, /*value=*/NULL, &size));
  char *kernel_name = new char[size];
  ASSERT_NO_CL_ERROR(
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, size, kernel_name, NULL));
  string s(kernel_name);
  delete[] kernel_name;
  return s;
}

void CLWrapper::run_kernel(cl_kernel kernel, 
  cl_uint work_dim,
  const size_t *global_work_size,
  const size_t *local_work_size,
  const size_t *global_work_offset,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list,
  cl_event *event) {
  ASSERT_NO_CL_ERROR(
    clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event));
}

float CLWrapper::run_kernel_with_timing(cl_kernel kernel,
  cl_uint work_dim,
  const size_t *global_work_size,
  const size_t *local_work_size,
  const size_t *global_work_offset,
  cl_uint num_events_in_wait_list,
  const cl_event *event_wait_list) {
  cl_event e;
  run_kernel(kernel,
    work_dim,
    global_work_size,
    local_work_size,
    global_work_offset,
    num_events_in_wait_list,
    event_wait_list,
    &e);
  if (profiling) {
    return time_and_release_event(e);
  } else {
    return 0;
  }
}

// --------------------------------------------------------------------------
// Private member functions
// --------------------------------------------------------------------------

float CLWrapper::timestamp_diff_in_ms(cl_ulong start, cl_ulong end) {
  return (end-start) * 1.0e-6f;
}

float CLWrapper::time_and_release_event(cl_event e) {
  cl_ulong start;
  cl_ulong end;
  ASSERT_NO_CL_ERROR(
    clWaitForEvents(/*num_events=*/1, &e));
  ASSERT_NO_CL_ERROR(
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, /*param_value_size_ret=*/NULL));
  ASSERT_NO_CL_ERROR(
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, /*param_value_size_ret=*/NULL));
  ASSERT_NO_CL_ERROR(clReleaseEvent(e));
  return timestamp_diff_in_ms(start, end);
}

void CLWrapper::attach_context(bool all_devices) {
  if (all_devices) {
    LOG(LOG_INFO, "Attaching context for all devices");
  } else {
    LOG(LOG_INFO, "Attaching context for device %d", d);
  }

  cl_uint ndev = (all_devices ? num_devices : 1);
  cl_device_id *dev = (all_devices ? devices : &devices[d]);
  cl_int ret;
  context = clCreateContext(/*properties=*/NULL, ndev, dev, context_error_callback, NULL, &ret);
  assert(context);
  ASSERT_NO_CL_ERROR(ret);
}

void CLWrapper::attach_command_queue(cl_command_queue_properties properties) {
  LOG(LOG_INFO, "Attaching command queue%sfor device %d", profiling ? " with profiling " : " ", d);
  cl_int ret;
  command_queue = clCreateCommandQueue(context, devices[d], properties, &ret);
  assert(command_queue);
  ASSERT_NO_CL_ERROR(ret);
}

void CLWrapper::set_kernel_arg(cl_kernel k, int i, int &n) {
  LOG(LOG_INFO, "Kernel arg%d of %s is int %d", i, name_of_kernel(k).c_str(), n);
  ASSERT_NO_CL_ERROR(
    clSetKernelArg(k, i, sizeof(int), &n));
}

void CLWrapper::set_kernel_arg(cl_kernel k, int i, float &n) {
  LOG(LOG_INFO, "Kernel arg%d of %s is float %f", i, name_of_kernel(k).c_str(), n);
  ASSERT_NO_CL_ERROR(
    clSetKernelArg(k, i, sizeof(float), &n));
}

void CLWrapper::set_kernel_arg(cl_kernel k, int i, double &n) {
  LOG(LOG_INFO, "Kernel arg%d of %s is double %f", i, name_of_kernel(k).c_str(), n);
  ASSERT_NO_CL_ERROR(
    clSetKernelArg(k, i, sizeof(double), &n));
}

void CLWrapper::set_kernel_arg(cl_kernel k, int i, cl_mem &m) {
  LOG(LOG_INFO, "Kernel arg%d of %s is cl_mem", i, name_of_kernel(k).c_str());
  ASSERT_NO_CL_ERROR(
    clSetKernelArg(k, i, sizeof(cl_mem), (void *)&m));
}

void CLWrapper::set_kernel_arg(cl_kernel k, int i, size_t &sz) {
  LOG(LOG_INFO, "Kernel arg%d of %s is local mem of size %lu", i, name_of_kernel(k).c_str(), sz);
  ASSERT_NO_CL_ERROR(
    clSetKernelArg(k, i, sz, NULL));
}

// --------------------------------------------------------------------------
// Helper functions
// --------------------------------------------------------------------------

cl_uint query_num_platforms() {
  static cl_uint num_platforms = 0;
  if (num_platforms == 0) {
    ASSERT_NO_CL_ERROR(
      clGetPlatformIDs(/*num_entries=*/0, /*platforms=*/NULL, &num_platforms));
  }
  return num_platforms;
}

cl_platform_id *&get_platform_list() {
  cl_uint num_platforms = query_num_platforms();
  static cl_platform_id *platforms = NULL;
  if (platforms == NULL) {
    platforms = new cl_platform_id[num_platforms];
    assert(platforms);
    ASSERT_NO_CL_ERROR(
      clGetPlatformIDs(num_platforms, platforms, /*num_platforms=*/NULL));
  }
  return platforms;
}

cl_uint query_num_devices(cl_platform_id platform) {
  cl_uint num_devices;
  ASSERT_NO_CL_ERROR(
    clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_ALL, /*num_entries=*/0, /*devices=*/NULL, &num_devices));
  return num_devices;
}

cl_device_id *get_device_list(cl_platform_id platform) {
  cl_uint num_devices = query_num_devices(platform);
  cl_device_id *devices = new cl_device_id[num_devices];
  ASSERT_NO_CL_ERROR(
    clGetDeviceIDs(
      platform, CL_DEVICE_TYPE_ALL, num_devices, devices, /*num_devices=*/NULL));
  return devices;
}

void context_error_callback(const char *errinfo, const void * /*unused*/, size_t /* unused */, void * /* unused */) {
  LOG(LOG_ERR, "context_error_callback %s", errinfo);
}

string clinfo() {
  stringstream ss;

  // PLATFORMS
  // query number of platforms
  cl_uint num_platforms = query_num_platforms();
  ss << "# Found " << num_platforms << " OpenCL platform" << (num_platforms == 1 ?  "":"s") << "\n";
  // get platform list
  cl_platform_id *platforms = get_platform_list();

  // query platform and devices
  char platform_name[1024];
  char platform_version[1024];
  char device_name[1024];
  char device_vendor[1024];
  cl_uint num_cores;
  cl_uint clk_freq;
  cl_long global_mem_size;
  cl_ulong local_mem_size;
  for (int i=0; i<(int)num_platforms; i++) {
    ASSERT_NO_CL_ERROR(
      clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, /*param_value_size_ret=*/NULL));
    ASSERT_NO_CL_ERROR(
      clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, /*param_value_size_ret=*/NULL));
    cl_uint num_devices = query_num_devices(platforms[i]);
    ss << "# Platform " << i << "\n";
    ss << "# Name: " << platform_name << "\n";
    ss << "# Version: " << platform_version << "\n";
    ss << "# Number of devices: " << num_devices << "\n";

    // get device list
    cl_device_id *devices = get_device_list(platforms[i]);
    for (int j=0; j<(int)num_devices; j++) {
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num_cores), &num_cores, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clk_freq), &clk_freq, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, /*param_value_size_ret=*/NULL));
      ASSERT_NO_CL_ERROR(
          clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, /*param_value_size_ret=*/NULL));

      ss << "# Device " << j << "\n";
      ss << "# \tName: " << device_name << "\n";
      ss << "# \tVendor: " << device_vendor << "\n";
      ss << "# \tCompute units: " << num_cores << "\n";
      ss << "# \tClock frequency: " << clk_freq << " MHz\n";
      ss << "# \tGlobal memory: " << (global_mem_size>>30) << "GB\n";
      ss << "# \tLocal memory: " <<  (local_mem_size>>10) << "KB\n";
    }
    delete[] devices;
  }

  return ss.str();
}
