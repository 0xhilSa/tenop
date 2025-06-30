#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include <cuComplex.h>


#define CUDA_ERROR(call)                                        \
  do{                                                           \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess){                                    \
      PyErr_Format(PyExc_RuntimeError,                          \
          "CUDA_ERROR: %s at %s:%d",                            \
          cudaGetErrorString(err), __FILE__, __LINE__);         \
      return nullptr;                                           \
    }                                                           \
  }while (0)

void cpu_free(PyObject *pycapsule){
  void *ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr) free(ptr);
}

void cuda_free(PyObject *pycapsule){
  void *ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(ptr) cudaFree(ptr);
}

static inline void *fail_exit(void *buffer){
  free(buffer);
  return NULL;
}

size_t get_size(const char *fmt){
  if(fmt == NULL) return 0;
  switch(*fmt){
    case '?': return sizeof(bool);
    case 'b': case 'B': return sizeof(char);
    case 'h': case 'H': return sizeof(short);
    case 'i': case 'I': return sizeof(int);
    case 'l': case 'L': return sizeof(long);
    case 'q': case 'Q': return sizeof(long long);
    case 'f': return sizeof(float);
    case 'd': case 'g': return sizeof(double);
    case 'F': return sizeof(cuFloatComplex);
    case 'D': case 'G': return sizeof(cuDoubleComplex);
    default: return 0;
  }
}

static PyObject *count(PyObject *self, PyObject *args){
  int count = 0;
  CUDA_ERROR(cudaGetDeviceCount(&count));
  return PyLong_FromLong(count);
}

void *__typecasting_list(PyObject *pylist, Py_ssize_t length, const char *fmt){
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return 0;
  }
  void *buffer = malloc(size * length);
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject *item = PyList_GetItem(pylist, i);
    void *dest = (char *)buffer + i * size;
    switch(*fmt){
      case '?': {
        bool val = (bool)PyObject_IsTrue(item);
        memcpy(dest, &val, size);
        break;
      }
      case 'b': {
        char val = (char)PyLong_AsLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'B': {
        unsigned char val = (unsigned char)PyLong_AsUnsignedLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'h': {
        short val = (short)PyLong_AsLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'H': {
        unsigned short val = (unsigned short)PyLong_AsUnsignedLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'i': {
        int val = (int)PyLong_AsLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'I': {
        unsigned int val = (unsigned int)PyLong_AsUnsignedLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'l': {
        long val = (long)PyLong_AsLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'L': {
        unsigned long val = (unsigned long)PyLong_AsUnsignedLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'q': {
        long long val = (long long)PyLong_AsLongLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'Q': {
        unsigned long long val = (unsigned long long)PyLong_AsUnsignedLongLong(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'f': {
        float val = (float)PyFloat_AsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'd': case 'g': {
        double val = (double)PyFloat_AsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'F': {
        float real = (float)PyComplex_RealAsDouble(item);
        float imag = (float)PyComplex_ImagAsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        cuFloatComplex val = make_cuFloatComplex(real, imag);
        memcpy(dest, &val, size);
        break;
      }
      case 'D': case 'G': {
        double real = (double)PyComplex_RealAsDouble(item);
        double imag = (double)PyComplex_ImagAsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        cuDoubleComplex val = make_cuDoubleComplex(real, imag);
        memcpy(dest, &val, size);
        break;
      }
      default: {
        free(buffer);
        PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
        return NULL;
      }
    }
  }
  return buffer;
}

void *__typecasting_scalar(PyObject *pyscalar, const char *fmt){
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return 0;
  }
  void *buffer = malloc(size);
  switch(*fmt){
    case '?': {
      bool val = (bool)PyObject_IsTrue(pyscalar);
      memcpy(buffer, &val, size);
      break;
    }
    case 'b': {
      char val = (char)PyLong_AsLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'B': {
      unsigned char val = (unsigned char)PyLong_AsUnsignedLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'h': {
      short val = (short)PyLong_AsLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'H': {
      unsigned short val = (unsigned short)PyLong_AsUnsignedLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'i': {
      int val = (int)PyLong_AsLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'I': {
      unsigned int val = (unsigned int)PyLong_AsUnsignedLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'l': {
      long val = (long)PyLong_AsLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'L': {
      unsigned long val = (unsigned long)PyLong_AsUnsignedLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'q': {
      long long val = (long long)PyLong_AsLongLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'Q': {
      unsigned long long val = (unsigned long long)PyLong_AsUnsignedLongLong(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'f': {
      float val = (float)PyFloat_AsDouble(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'd': case 'g': {
      double val = (double)PyFloat_AsDouble(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'F': {
      float real = (float)PyComplex_RealAsDouble(pyscalar);
      float imag = (float)PyComplex_ImagAsDouble(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      cuFloatComplex val = make_cuFloatComplex(real, imag);
      memcpy(buffer, &val, size);
      break;
    }
    case 'D': case 'G': {
      double real = (double)PyComplex_RealAsDouble(pyscalar);
      double imag = (double)PyComplex_ImagAsDouble(pyscalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      cuDoubleComplex val = make_cuDoubleComplex(real, imag);
      memcpy(buffer, &val, size);
      break;
    }
    default: {
      free(buffer);
      PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
      return NULL;
    }
  }
  return buffer;
}

static PyObject* get_cuda_device_prop(PyObject* self, PyObject* args) {
  int device_id = 0;
  if(!PyArg_ParseTuple(args, "|i", &device_id)) return NULL;

  int device_count = 0;
  if(cudaGetDeviceCount(&device_count) != cudaSuccess || device_id >= device_count) return PyErr_Format(PyExc_RuntimeError, "Invalid or unavailable CUDA device.");

  cudaDeviceProp prop;
  if(cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) return PyErr_Format(PyExc_RuntimeError, "Failed to get device properties.");

  PyObject* dict = PyDict_New();

  PyDict_SetItemString(dict, "name", PyUnicode_FromString(prop.name));
  PyDict_SetItemString(dict, "total_global_mem", PyLong_FromUnsignedLongLong(prop.totalGlobalMem));
  PyDict_SetItemString(dict, "compute_capability", Py_BuildValue("(ii)", prop.major, prop.minor));
  PyDict_SetItemString(dict, "multi_processor_count", PyLong_FromLong(prop.multiProcessorCount));
  PyDict_SetItemString(dict, "max_threads_per_block", PyLong_FromLong(prop.maxThreadsPerBlock));

  PyObject* block_dim = Py_BuildValue("(iii)",
      prop.maxThreadsDim[0],
      prop.maxThreadsDim[1],
      prop.maxThreadsDim[2]
  );
  PyDict_SetItemString(dict, "max_block_dim", block_dim);
  Py_DECREF(block_dim);

  PyObject* grid_dim = Py_BuildValue("(iii)",
      prop.maxGridSize[0],
      prop.maxGridSize[1],
      prop.maxGridSize[2]
  );
  PyDict_SetItemString(dict, "max_grid_dim", grid_dim);
  Py_DECREF(grid_dim);
  return dict;
}

static void *__cpu2cuda(PyObject *pycapsule, int numel, int device_id, const char *fmt){
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return 0;
  }
  CUDA_ERROR(cudaSetDevice(device_id));
  void *buffer = PyCapsule_GetPointer(pycapsule, "CPU");
  void *cuda_buffer;
  CUDA_ERROR(cudaMalloc(&cuda_buffer, size * numel));
  CUDA_ERROR(cudaMemcpy(cuda_buffer, buffer, size * numel, cudaMemcpyHostToDevice));
  return cuda_buffer;
}

static void *__cuda2cpu(PyObject *cuda_capsule, int numel, const char *fmt){
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return NULL;
  }
  void *device_buffer = PyCapsule_GetPointer(cuda_capsule, "CUDA");
  if(!device_buffer){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA capsule or null pointer.");
    return NULL;
  }
  void *host_buffer = malloc(size * numel);
  if(!host_buffer){
    PyErr_NoMemory();
    return NULL;
  }
  CUDA_ERROR(cudaMemcpy(host_buffer, device_buffer, size * numel, cudaMemcpyDeviceToHost));
  return host_buffer;
}

static void *__list2cuda(PyObject *pylist, int device_id, const char *fmt){
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return 0;
  }
  Py_ssize_t numel = PyList_Size(pylist);
  void *host_buffer = __typecasting_list(pylist, numel, fmt);
  void *device_buffer;
  CUDA_ERROR(cudaMalloc(&device_buffer, size * numel));
  CUDA_ERROR(cudaMemcpy(device_buffer, host_buffer, size * numel, cudaMemcpyHostToDevice));
  return device_buffer;
}

static void *__scalar2cuda(PyObject *pyscalar, int device_ptr, const char *fmt){
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return 0;
  }
  void *host_buffer = __typecasting_scalar(pyscalar, fmt);
  void *device_buffer;
  CUDA_ERROR(cudaMalloc(&device_buffer, size));
  CUDA_ERROR(cudaMemcpy(device_buffer, host_buffer, size, cudaMemcpyHostToDevice));
  return device_buffer;
}

static PyObject *tocuda(PyObject *self, PyObject *args){
  PyObject *pyobject;
  int device_id, numel;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "Oiis", &pyobject, &numel, &device_id, &fmt)) return NULL;
  void *device_ptr;
  if(PyCapsule_CheckExact(pyobject)){
    device_ptr = __cpu2cuda(pyobject, numel, device_id, fmt);
  }else if(PyList_Check(pyobject)){
    device_ptr = __list2cuda(pyobject, device_id, fmt);
  }else if(PyNumber_Check(pyobject) || PyComplex_Check(pyobject)){
    device_ptr = __scalar2cuda(pyobject, device_id, fmt);
  }else{
    PyErr_SetString(PyExc_TypeError, "Unsupported input type. Must be scalar, list, or CPU PyCapsule.");
    return NULL;
  }
  if(device_ptr == NULL) return NULL;
  return PyCapsule_New(device_ptr, "CUDA", cuda_free);
}

static PyObject *tocpu(PyObject *self, PyObject *args){
  PyObject *pycapsule;
  int numel;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "Ois", &pycapsule, &numel, &fmt)) return NULL;
  void *host_buffer = __cuda2cpu(pycapsule, numel, fmt);
  if(!host_buffer) return NULL;
  return PyCapsule_New(host_buffer, "CPU", cpu_free);
}

static PyObject *device_name(PyObject *self, PyObject *args){
  int device_id;
  if(!PyArg_ParseTuple(args, "i", &device_id)) return NULL;
  cudaDeviceProp prop;
  CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));
  return Py_BuildValue("s", prop.name);
}

static PyObject *current_device(PyObject* self, PyObject *args){
  int device_id;
  CUDA_ERROR(cudaGetDevice(&device_id));
  return Py_BuildValue("i", device_id);
}

static PyObject *get_arch_list(PyObject *self, PyObject *args){
  int device_count;
  CUDA_ERROR(cudaGetDeviceCount(&device_count));
  PyObject *list = PyList_New(0);
  for(int i = 0; i < device_count; ++i){
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, i);
    if(err != cudaSuccess){
      PyErr_Format(PyExc_RuntimeError, "cudaGetDeviceProperties failed for device %d: %s", i, cudaGetErrorString(err));
      Py_DECREF(list);
      return NULL;
    }
    PyObject *arch_tuple = Py_BuildValue("(ii)", prop.major, prop.minor);
    PyList_Append(list, arch_tuple);
    Py_DECREF(arch_tuple);
  }
  return list;
}

static PyMethodDef CudaMethods[] = {
  {"count", count, METH_NOARGS, "Returns number of CUDA devices"},
  {"get_prop", get_cuda_device_prop, METH_VARARGS, "get CUDA properties"},
  {"tocuda", tocuda, METH_VARARGS, "to CUDA device"},
  {"tocpu", tocpu, METH_VARARGS, "to CPU"},
  {"get_device_name", device_name, METH_VARARGS, "get CUDA device name"},
  {"current_device", current_device, METH_NOARGS, "get current CUDA device"},
  {"get_arch_list", get_arch_list, METH_NOARGS, "get arch lists"},
  {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef cudamodule = {
  PyModuleDef_HEAD_INIT,
  "cuda",
  nullptr,
  -1,
  CudaMethods
};

PyMODINIT_FUNC PyInit_cuda(void) {
  return PyModule_Create(&cudamodule);
}
