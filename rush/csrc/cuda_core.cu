#include <python3.10/Python.h>
#include <cuComplex.h>
#include <stdbool.h>

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

static size_t sizeofs(const char* fmt){
  if(fmt == NULL){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return 0;
  }
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
    case 'F': return sizeof(cuFloatComplex);
    case 'D': case 'G': return sizeof(cuDoubleComplex);
    default:  return 0;
  }
}

static void cuda_free(PyObject* pycapsule){
  void* device_ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(device_ptr) cudaFree(device_ptr);
}

static void cpu_free(PyObject* pycapsule){
  void* host_ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(host_ptr) free(host_ptr);
}

static PyObject* to_cuda(PyObject* self, PyObject* args){
  PyObject* pylist;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pylist, &fmt)) return NULL;
  if(!PyList_Check(pylist)){
    PyErr_SetString(PyExc_TypeError, "Expected a Python List");
    return NULL;
  }
  size_t length = (size_t)PyList_Size(pylist);
  size_t size = sizeofs(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  size_t total_size = length * size;
  void* buffer = malloc(total_size);
  for(size_t i = 0; i < length; i++){
    PyObject* item = PyList_GET_ITEM(pylist, i);
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
      case 'D': {
                  double real = (double)PyComplex_RealAsDouble(item);
                  double imag = (double)PyComplex_ImagAsDouble(item);
                  ((cuDoubleComplex*)buffer)[i] = make_cuDoubleComplex(real, imag);
                  break;
                }
      case 'G': {
                  double real = (double)PyComplex_RealAsDouble(item);
                  double imag = (double)PyComplex_ImagAsDouble(item);
                  ((cuDoubleComplex*)buffer)[i] = make_cuDoubleComplex(real, imag);
                  break;
                }
      default:  {
                  free(buffer);
                  PyErr_SetString(PyExc_ValueError, "Type Conversion Failed :(");
                  return NULL;
                }
    }
  }
  void* device_ptr;
  CUDA_ERROR(cudaMalloc(&device_ptr, total_size));
  CUDA_ERROR(cudaMemcpy(device_ptr, buffer, total_size, cudaMemcpyHostToDevice));
  return PyCapsule_New(device_ptr, "CUDA", cuda_free);
}

static PyObject* to_cpu(PyObject* self, PyObject* args){
  PyObject* pycapsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &pycapsule, &length, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "pycapsule must be a pointer");
    return NULL;
  }
  size_t size = sizeofs(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(!device_ptr) return NULL;
  void* host_ptr = malloc((size_t)length * size);
  if(!host_ptr){
    PyErr_NoMemory();
    return NULL;
  }
  size_t total_size = (size_t)length * size;
  CUDA_ERROR(cudaMemcpy(host_ptr, device_ptr, total_size, cudaMemcpyDeviceToHost));
  return PyCapsule_New(host_ptr, "CPU", cpu_free);
}

static PyObject* to_list(PyObject* self, PyObject* args){
  PyObject* pycapsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &pycapsule, &length, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "pycapsule must be a pointer");
    return NULL;
  }
  size_t size = sizeofs(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  void* device_ptr = PyCapsule_GetPointer(pycapsule, "CUDA");
  if(!device_ptr) return NULL;
  void* buffer = malloc((size_t)length * size);
  if(!buffer){
    PyErr_NoMemory();
    return NULL;
  }
  CUDA_ERROR(cudaMemcpy(buffer, device_ptr, (size_t)length * size, cudaMemcpyDeviceToHost));
  PyObject* pylist = PyList_New(length);
  if(!pylist){
    free(buffer);
    return NULL;
  }
  for(Py_ssize_t i = 0; i < length; ++i){
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
      case 'g': item = PyFloat_FromDouble((double)((long double*)buffer)[i]); break;
      case 'F': {
                  cuFloatComplex c = ((cuFloatComplex*)buffer)[i];
                  item = PyComplex_FromDoubles((double)cuCrealf(c), (double)cuCimagf(c));
                  break;
                }
        case 'D': case 'G': {
                              cuDoubleComplex c = ((cuDoubleComplex*)buffer)[i];
                              item = PyComplex_FromDoubles(cuCreal(c), cuCimag(c));
                              break;
                            }
        default:  {
                    free(buffer);
                    Py_DECREF(pylist);
                    PyErr_SetString(PyExc_ValueError, "Unsupported format during decode");
                    return NULL;
                  }
    }
    if(!item){
      free(buffer);
      Py_DECREF(pylist);
      return NULL;
    }
    PyList_SET_ITEM(pylist, i, item);
  }
  free(buffer);
  return pylist;
}

static PyMethodDef methods[] = {
  {"toCUDA", to_cuda, METH_VARARGS, "Move host memory to CUDA device memory"},
  {"toCPU", to_cpu, METH_VARARGS, "Move CUDA device memory to host memory"},
  {"toList", to_list, METH_VARARGS, "Return PyList from CUDA device memory"},
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
