#include <cuda_runtime.h>
#include <python3.10/Python.h>
#include <cuComplex.h>
#include <python3.10/pycapsule.h>
#include <stdbool.h>


void cuda_free(PyObject *pyobject){
  void *ptr = PyCapsule_GetPointer(pyobject, "CUDA");
  if(ptr) cudaFree(ptr);
}

