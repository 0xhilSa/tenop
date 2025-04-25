#include <python3.10/Python.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <python3.10/methodobject.h>
#include <python3.10/pycapsule.h>
#include <python3.10/pyerrors.h>

#define CUDA_ERROR(call)                                 \
  do{                                                    \
    cudaError_t err = call;                              \
    if (err != cudaSuccess){                             \
      PyErr_Format(PyExc_RuntimeError,                   \
          "CUDA_ERROR: %s at %s:%d",                     \
          cudaGetErrorString(err), __FILE__, __LINE__);  \
      return nullptr;                                    \
    }                                                    \
  }while (0)

static size_t get_size(const char* fmt){
  if(fmt == NULL) return -1;
  switch(*fmt){
    case '?': return sizeof(bool);
    case 'b': case 'B': return sizeof(char);
    case 'h': case 'H': return sizeof(short);
    case 'i': case 'I': return sizeof(int);
    case 'l': case 'L': return sizeof(long);
    case 'q': case 'Q': return sizeof(long long);
    case 'f': return sizeof(float);
    case 'd': return sizeof(double);
    case 'g': return sizeof(long double);
    case 'F': return sizeof(float _Complex);
    case 'D': return sizeof(double _Complex);
    case 'G': return sizeof(long double _Complex);
    default:  return 0;
  }
}

static void cuda_free(PyObject* pycapsule){
  void* ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(ptr) cudaFree(ptr);
}

static void cpu_free(PyObject* pycapsule){
  void* ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr) free(ptr);
}

static PyObject* s_tocuda(PyObject* self, PyObject* args){
  PyObject* scalar;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &scalar, &fmt)) return NULL;
  size_t size = get_size(fmt);
  if(size == (size_t)-1){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return NULL;
  }else if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  void* buffer = malloc(size);
  switch(*fmt){
    case '?': *((bool*)buffer) = (bool)PyObject_IsTrue(scalar); break;
    case 'b': *((char*)buffer) = (char)PyLong_AsLong(scalar); break;
    case 'B': *((unsigned char*)buffer) = (unsigned char)PyLong_AsUnsignedLong(scalar); break;
    case 'h': *((short*)buffer) = (short)PyLong_AsLong(scalar); break;
    case 'H': *((unsigned short*)buffer) = (unsigned short)PyLong_AsUnsignedLong(scalar); break;
    case 'i': *((int*)buffer) = (int)PyLong_AsLong(scalar); break;
    case 'I': *((unsigned int*)buffer) = (unsigned int)PyLong_AsUnsignedLong(scalar); break;
    case 'l': *((long*)buffer) = (long)PyLong_AsLong(scalar); break;
    case 'L': *((unsigned long*)buffer) = (unsigned long)PyLong_AsUnsignedLong(scalar); break;
    case 'q': *((long long*)buffer) = (long long)PyLong_AsLongLong(scalar); break;
    case 'Q': *((unsigned long long*)buffer) = (unsigned long long)PyLong_AsUnsignedLongLong(scalar); break;
    case 'f': *((float*)buffer) = (float)PyFloat_AsDouble(scalar); break;
    case 'd': *((double*)buffer) = (double)PyFloat_AsDouble(scalar); break;
    case 'g': *((long double*)buffer) = (long double)PyFloat_AsDouble(scalar); break;
    case 'F': {
      float real = (float)PyComplex_RealAsDouble(scalar);
      float imag = (float)PyComplex_ImagAsDouble(scalar);
      *((cuFloatComplex*)buffer) = make_cuFloatComplex(real, imag);
      break;
    }
    case 'D': case 'G': {
      double real = (double)PyComplex_RealAsDouble(scalar);
      double imag = (double)PyComplex_ImagAsDouble(scalar);
      *((cuDoubleComplex*)buffer) = make_cuDoubleComplex(real, imag);
      break;
    }
    default:  {
      free(buffer);
      PyErr_Format(PyExc_ValueError, "Unsupported format character: %c", *fmt);
      return NULL;
    }
  }
  void* device_ptr;
  CUDA_ERROR(cudaMalloc(&device_ptr, size));
  CUDA_ERROR(cudaMemcpy(device_ptr, buffer, size, cudaMemcpyHostToDevice));
  return PyCapsule_New(device_ptr, "CUDA", cuda_free);
}

static PyObject* s_fromcuda(PyObject* self, PyObject* args){
  PyObject* pycapsule;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pycapsule, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "Invalid PyCapsule/Pointer");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid or NULL capsule pointer");
    return NULL;
  }
  size_t size = get_size(fmt);
  if(size == (size_t)-1){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return NULL;
  } else if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  void* host_buffer = malloc(size);
  if(!host_buffer){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host buffer");
    return NULL;
  }
  CUDA_ERROR(cudaMemcpy(host_buffer, device_ptr, size, cudaMemcpyDeviceToHost));
  switch(*fmt){
    case '?': return PyBool_FromLong(*(bool*)host_buffer);
    case 'b': return PyLong_FromLong(*(char*)host_buffer);
    case 'B': return PyLong_FromUnsignedLong(*(unsigned char*)host_buffer);
    case 'h': return PyLong_FromLong(*(short*)host_buffer);
    case 'H': return PyLong_FromUnsignedLong(*(unsigned short*)host_buffer);
    case 'i': return PyLong_FromLong(*(int*)host_buffer);
    case 'I': return PyLong_FromUnsignedLong(*(unsigned int*)host_buffer);
    case 'l': return PyLong_FromLong(*(long*)host_buffer);
    case 'L': return PyLong_FromUnsignedLong(*(unsigned long*)host_buffer);
    case 'q': return PyLong_FromLongLong(*(long long*)host_buffer);
    case 'Q': return PyLong_FromUnsignedLongLong(*(unsigned long long*)host_buffer);
    case 'f': return PyFloat_FromDouble(*(float*)host_buffer);
    case 'd': return PyFloat_FromDouble(*(double*)host_buffer);
    case 'g': return PyFloat_FromDouble(*(long double*)host_buffer);
    case 'F': {
      cuFloatComplex val = *(cuFloatComplex*)host_buffer;
      return PyComplex_FromDoubles(cuCrealf(val), cuCimagf(val));
    }
    case 'D': {
      cuDoubleComplex val = *(cuDoubleComplex*)host_buffer;
      return PyComplex_FromDoubles(cuCreal(val), cuCimag(val));
    }
    case 'G': {
      cuDoubleComplex val = *(cuDoubleComplex*)host_buffer;
      return PyComplex_FromDoubles(cuCreal(val), cuCimag(val));
    }
    default: {
      free(host_buffer);
      PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
      return NULL;
    }
  }
  free(host_buffer);
}

static PyObject* s_tohost(PyObject* self, PyObject* args){
  PyObject* pycapsule;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pycapsule, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "Expected a PyCapsule for CUDA memory");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid or NULL CUDA capsule pointer");
    return NULL;
  }
  size_t size = get_size(fmt);
  if(size == (size_t)-1){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return NULL;
  } else if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  void* host_ptr = malloc(size);
  if(!host_ptr){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host memory");
    return NULL;
  }
  CUDA_ERROR(cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost));
  return PyCapsule_New(host_ptr, "CPU", cpu_free);
}

static PyObject* v_tocuda(PyObject* self, PyObject* args){
  PyObject* pylist;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pylist, &fmt)) return NULL;
  if(!PyList_Check(pylist)){
    PyErr_SetString(PyExc_TypeError, "Expected a Python list");
    return NULL;
  }
  Py_ssize_t length = PyList_Size(pylist);
  size_t size = get_size(fmt);
  if(size == (size_t)-1){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return NULL;
  }else if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  size_t total_bytes = (size_t)length * size;
  void* buffer = malloc(total_bytes);
  if(!buffer){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host buffer");
    return NULL;
  }
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject* item = PyList_GetItem(pylist, i);
    if(!item){
      free(buffer);
      PyErr_Format(PyExc_ValueError, "List item %zd is NULL", i);
      return NULL;
    }
    switch(*fmt){
      case '?': ((bool*)buffer)[i] = (bool)PyObject_IsTrue(item); break;
      case 'b': ((char*)buffer)[i] = (char)PyLong_AsLong(item); break;
      case 'B': ((unsigned char*)buffer)[i] = (unsigned char)PyLong_AsUnsignedLong(item); break;
      case 'h': ((short*)buffer)[i] = (short)PyLong_AsLong(item); break;
      case 'H': ((unsigned short*)buffer)[i] = (unsigned short)PyLong_AsUnsignedLong(item); break;
      case 'i': ((int*)buffer)[i] = (int)PyLong_AsLong(item); break;
      case 'I': ((unsigned int*)buffer)[i] = (unsigned int)PyLong_AsUnsignedLong(item); break;
      case 'l': ((long*)buffer)[i] = (long)PyLong_AsLong(item); break;
      case 'L': ((unsigned long*)buffer)[i] = (unsigned long)PyLong_AsUnsignedLong(item); break;
      case 'q': ((long long*)buffer)[i] = (long long)PyLong_AsLongLong(item); break;
      case 'Q': ((unsigned long long*)buffer)[i] = (unsigned long long)PyLong_AsUnsignedLongLong(item); break;
      case 'f': ((float*)buffer)[i] = (float)PyFloat_AsDouble(item); break;
      case 'd': ((double*)buffer)[i] = (double)PyFloat_AsDouble(item); break;
      case 'g': ((long double*)buffer)[i] = (long double)PyFloat_AsDouble(item); break;
      case 'F': {
        float real = (float)PyComplex_RealAsDouble(item);
        float imag = (float)PyComplex_ImagAsDouble(item);
        ((cuFloatComplex*)buffer)[i] = make_cuFloatComplex(real, imag);
        break;
      }
      case 'D': case 'G': {
        double real = (double)PyComplex_RealAsDouble(item);
        double imag = (double)PyComplex_ImagAsDouble(item);
        ((cuDoubleComplex*)buffer)[i] = make_cuDoubleComplex(real, imag);
        break;
      }
      default: {
        free(buffer);
        PyErr_Format(PyExc_ValueError, "Unsupported format character: %c", *fmt);
        return NULL;
      }
    }
    if(PyErr_Occurred()){
      free(buffer);
      return NULL;
    }
  }
  void* device_ptr;
  CUDA_ERROR(cudaMalloc(&device_ptr, total_bytes));
  CUDA_ERROR(cudaMemcpy(device_ptr, buffer, total_bytes, cudaMemcpyHostToDevice));
  free(buffer);
  return PyCapsule_New(device_ptr, "CUDA", cuda_free);
}

static PyObject* v_fromcuda(PyObject* self, PyObject* args){
  PyObject* pycapsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &pycapsule, &length, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "Expected a CUDA PyCapsule");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA pointer");
    return NULL;
  }
  size_t size = get_size(fmt);
  if(size == (size_t)-1){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return NULL;
  }else if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  size_t total_bytes = length * size;
  void* buffer = malloc(total_bytes);
  if(!buffer){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host buffer");
    return NULL;
  }
  CUDA_ERROR(cudaMemcpy(buffer, device_ptr, total_bytes, cudaMemcpyDeviceToHost));
  PyObject* pylist = PyList_New(length);
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject* item = NULL;
    switch(*fmt){
      case '?': item = PyBool_FromLong(((bool*)buffer)[i]); break;
      case 'b': item = PyLong_FromLong(((char*)buffer)[i]); break;
      case 'B': item = PyLong_FromUnsignedLong(((unsigned char*)buffer)[i]); break;
      case 'h': item = PyLong_FromLong(((short*)buffer)[i]); break;
      case 'H': item = PyLong_FromUnsignedLong(((unsigned short*)buffer)[i]); break;
      case 'i': item = PyLong_FromLong(((int*)buffer)[i]); break;
      case 'I': item = PyLong_FromUnsignedLong(((unsigned int*)buffer)[i]); break;
      case 'l': item = PyLong_FromLong(((long*)buffer)[i]); break;
      case 'L': item = PyLong_FromUnsignedLong(((unsigned long*)buffer)[i]); break;
      case 'q': item = PyLong_FromLongLong(((long long*)buffer)[i]); break;
      case 'Q': item = PyLong_FromUnsignedLongLong(((unsigned long long*)buffer)[i]); break;
      case 'f': item = PyFloat_FromDouble(((float*)buffer)[i]); break;
      case 'd': item = PyFloat_FromDouble(((double*)buffer)[i]); break;
      case 'g': item = PyFloat_FromDouble(((long double*)buffer)[i]); break;
      case 'F': {
        cuFloatComplex val = ((cuFloatComplex*)buffer)[i];
        item = PyComplex_FromDoubles(cuCrealf(val), cuCimagf(val));
        break;
      }
      case 'D': case 'G': {
        cuDoubleComplex val = ((cuDoubleComplex*)buffer)[i];
        item = PyComplex_FromDoubles(cuCreal(val), cuCimag(val));
        break;
      }
      default: {
        free(buffer);
        PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
        return NULL;
      }
    }
    if(!item || PyErr_Occurred()){
      free(buffer);
      return NULL;
    }
    PyList_SetItem(pylist, i, item);
  }
  free(buffer);
  return pylist;
}

static PyObject* v_tohost(PyObject* self, PyObject* args){
  PyObject* pycapsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &pycapsule, &length, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "Expected a PyCapsule/Pointer");
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(!device_ptr){
    PyErr_SetString(PyExc_ValueError, "Invalid CUDA pointer");
    return NULL;
  }
  size_t size = get_size(fmt);
  if(size == (size_t)-1){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return NULL;
  }else if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  size_t total_bytes = (size_t)length * size;
  void* host_buffer = malloc(total_bytes);
  if(!host_buffer){
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate host buffer");
    return NULL;
  }
  CUDA_ERROR(cudaMemcpy(host_buffer, device_ptr, total_bytes, cudaMemcpyDeviceToHost));
  return PyCapsule_New(host_buffer, "CPU", cpu_free);
}

static PyMethodDef methods[] = {
  {"s_toCUDA", s_tocuda, METH_VARARGS, "put the scalar value on CUDA memory"},
  {"s_fromCUDA", s_fromcuda, METH_VARARGS, "return that scalar"},
  {"s_toHost", s_tohost, METH_VARARGS, "put that scalar value into the CPU memory"},
  {"v_toCUDA", v_tocuda, METH_VARARGS, "put the vector on CUDA memory"},
  {"v_fromCUDA", v_fromcuda, METH_VARARGS, "return the pylist from the CUDA memory"},
  {"v_toHost", v_tohost, METH_VARARGS, "put the vector from CUDA memory to CPU memory"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cuda_core",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cuda_core(void){
  return PyModule_Create(&module);
}
