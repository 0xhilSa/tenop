#include <python3.10/Python.h>
#include <cuda_runtime.h>


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


static PyObject *count(PyObject *self, PyObject *args){
  int count = 0;
  CUDA_ERROR(cudaGetDeviceCount(&count));
  return PyLong_FromLong(count);
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

static PyMethodDef CudaMethods[] = {
  {"count", count, METH_NOARGS, "Returns number of CUDA devices"},
  {"get_prop", get_cuda_device_prop, METH_VARARGS, "get CUDA properties"},
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
