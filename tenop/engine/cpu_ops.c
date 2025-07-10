#include <python3.10/Python.h>
#include <limits.h>
#include <float.h>
#include <python3.10/pyerrors.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>


void cpu_free(PyObject *pycapsule){
  void *ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr) free(ptr);
}

static void *reduced_max_kernel_bool(void *ptr, Py_ssize_t length){
  bool *buf = calloc(1, sizeof(bool));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for bool dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((bool *)ptr)[i] > *buf){ *buf = ((bool*)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_char(void *ptr, Py_ssize_t length){
  char *buf = calloc(1, sizeof(char));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for char dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((char *)ptr)[i] > *buf){ *buf = ((char *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_uchar(void *ptr, Py_ssize_t length){
  unsigned char *buf = calloc(1, sizeof(unsigned char));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned char dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned char *)ptr)[i] > *buf){ *buf = ((unsigned char *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_short(void *ptr, Py_ssize_t length){
  short *buf = calloc(1, sizeof(short));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for short dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((short *)ptr)[i] > *buf){ *buf = ((short *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_ushort(void *ptr, Py_ssize_t length){
  unsigned short *buf = calloc(1, sizeof(unsigned short));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned short dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned short *)ptr)[i] > *buf){ *buf = ((unsigned short *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_int(void *ptr, Py_ssize_t length){
  int *buf = calloc(1, sizeof(int));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for int dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((int *)ptr)[i] > *buf){ *buf = ((int *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_uint(void *ptr, Py_ssize_t length){
  unsigned int *buf = calloc(1, sizeof(unsigned int));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned int dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned int *)ptr)[i] > *buf){ *buf = ((unsigned int *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_long(void *ptr, Py_ssize_t length){
  long *buf = calloc(1, sizeof(long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for long dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long *)ptr)[i] > *buf){ *buf = ((long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_ulong(void *ptr, Py_ssize_t length){
  unsigned long *buf = calloc(1, sizeof(unsigned long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned long dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned long *)ptr)[i] > *buf){ *buf = ((unsigned long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_longlong(void *ptr, Py_ssize_t length){
  long long *buf = calloc(1, sizeof(long long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for long long dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long long *)ptr)[i] > *buf){ *buf = ((long long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_ulonglong(void *ptr, Py_ssize_t length){
  unsigned long long *buf = calloc(1, sizeof(unsigned long long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned long dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned long long *)ptr)[i] > *buf){ *buf = ((unsigned long long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_float(void *ptr, Py_ssize_t length){
  float *buf = calloc(1, sizeof(float));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for float dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((float *)ptr)[i] > *buf){ *buf = ((float *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_double(void *ptr, Py_ssize_t length){
  double *buf = calloc(1, sizeof(double));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for double dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((double *)ptr)[i] > *buf){ *buf = ((double *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_max_kernel_longdouble(void *ptr, Py_ssize_t length){
  long double *buf = calloc(1, sizeof(long double));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for long double dtype"); return NULL; }
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long double *)ptr)[i] > *buf){ *buf = ((long double *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_bool(void *ptr, Py_ssize_t length){
  bool *buf = malloc(sizeof(bool));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for bool dtype"); return NULL; }
  *buf = ((bool *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((bool *)ptr)[i] < *buf){ *buf = ((bool *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_char(void *ptr, Py_ssize_t length){
  char *buf = malloc(sizeof(char));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for char dtype"); return NULL; }
  *buf = ((char *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((char *)ptr)[i] < *buf){ *buf = ((char *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_uchar(void *ptr, Py_ssize_t length){
  unsigned char *buf = malloc(sizeof(unsigned char));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned char dtype"); return NULL; }
  *buf = ((unsigned char *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned char *)ptr)[i] < *buf){ *buf = ((unsigned char *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_short(void *ptr, Py_ssize_t length){
  short *buf = malloc(sizeof(short));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for short dtype"); return NULL; }
  *buf = ((short *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((short *)ptr)[i] < *buf){ *buf = ((short *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_ushort(void *ptr, Py_ssize_t length){
  unsigned short *buf = malloc(sizeof(unsigned short));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned short dtype"); return NULL; }
  *buf = ((unsigned short *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned short *)ptr)[i] < *buf){ *buf = ((unsigned short *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_int(void *ptr, Py_ssize_t length){
  int *buf = malloc(sizeof(int));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for int dtype"); return NULL; }
  *buf = ((int *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((int *)ptr)[i] < *buf){ *buf = ((int *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_uint(void *ptr, Py_ssize_t length){
  unsigned int *buf = malloc(sizeof(unsigned int));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned int dtype"); return NULL; }
  *buf = ((unsigned int *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned int *)ptr)[i] < *buf){ *buf = ((unsigned int *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_long(void *ptr, Py_ssize_t length){
  long *buf = malloc(sizeof(long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for long dtype"); return NULL; }
  *buf = ((long *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long *)ptr)[i] < *buf){ *buf = ((long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_ulong(void *ptr, Py_ssize_t length){
  unsigned long *buf = malloc(sizeof(unsigned long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned long dtype"); return NULL; }
  *buf = ((unsigned long *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned long *)ptr)[i] < *buf){ *buf = ((unsigned long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_longlong(void *ptr, Py_ssize_t length){
  long long *buf = malloc(sizeof(long long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for long long dtype"); return NULL; }
  *buf = ((long long *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long long *)ptr)[i] < *buf){ *buf = ((long long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_ulonglong(void *ptr, Py_ssize_t length){
  unsigned long long *buf = malloc(sizeof(unsigned long long));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for unsigned long long dtype"); return NULL; }
  *buf = ((unsigned long long *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned long long *)ptr)[i] < *buf){ *buf = ((unsigned long long *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_float(void *ptr, Py_ssize_t length){
  float *buf = malloc(sizeof(float));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for float dtype"); return NULL; }
  *buf = ((float *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((float *)ptr)[i] < *buf){ *buf = ((float *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_double(void *ptr, Py_ssize_t length){
  double *buf = malloc(sizeof(double));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for double dtype"); return NULL; }
  *buf = ((double *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((double *)ptr)[i] < *buf){ *buf = ((double *)ptr)[i]; }
  }
  return buf;
}

static void *reduced_min_kernel_longdouble(void *ptr, Py_ssize_t length){
  long double *buf = malloc(sizeof(long double));
  if(!buf){ PyErr_SetString(PyExc_MemoryError, "Failed to allocate the memory for long double dtype"); return NULL; }
  *buf = ((long double *)ptr)[0];
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long double *)ptr)[i] < *buf){ *buf = ((long double *)ptr)[i]; }
  }
  return buf;
}

static PyObject *reduced_max(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "Ons", &tensor, &length, &fmt)) return NULL;
  void *ptr = PyCapsule_GetPointer(tensor, "CPU");
  if(!ptr){
    PyErr_SetString(PyExc_RuntimeError, "Invalid Pointer");
    return NULL;
  }
  void *buf = NULL;
  switch(*fmt){
    case '?': { buf = reduced_max_kernel_bool(ptr, length); break; }
    case 'b': { buf = reduced_max_kernel_char(ptr, length); break; }
    case 'B': { buf = reduced_max_kernel_uchar(ptr, length); break; }
    case 'h': { buf = reduced_max_kernel_short(ptr, length); break; }
    case 'H': { buf = reduced_max_kernel_ushort(ptr, length); break; }
    case 'i': { buf = reduced_max_kernel_int(ptr, length); break; }
    case 'I': { buf = reduced_max_kernel_uint(ptr, length); break; }
    case 'l': { buf = reduced_max_kernel_long(ptr, length); break; }
    case 'L': { buf = reduced_max_kernel_ulong(ptr, length); break; }
    case 'q': { buf = reduced_max_kernel_longlong(ptr, length); break; }
    case 'Q': { buf = reduced_max_kernel_ulonglong(ptr, length); break; }
    case 'f': { buf = reduced_max_kernel_float(ptr, length); break; }
    case 'd': { buf = reduced_max_kernel_double(ptr, length); break; }
    case 'g': { buf = reduced_max_kernel_longdouble(ptr, length); break; }
    default: {
      PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
      return NULL;
    }
  }
  if(!buf){
    PyErr_SetString(PyExc_RuntimeError, "Kernel failed");
    return NULL;
  }
  return PyCapsule_New(buf, "CPU", cpu_free);
}

static PyObject *reduced_min(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "Ons", &tensor, &length, &fmt)) return NULL;
  void *ptr = PyCapsule_GetPointer(tensor, "CPU");
  if(!ptr){
    PyErr_SetString(PyExc_RuntimeError, "Invalid Pointer");
    return NULL;
  }
  void *buf = NULL;
  switch(*fmt){
    case '?': { buf = reduced_min_kernel_bool(ptr, length); break; }
    case 'b': { buf = reduced_min_kernel_char(ptr, length); break; }
    case 'B': { buf = reduced_min_kernel_uchar(ptr, length); break; }
    case 'h': { buf = reduced_min_kernel_short(ptr, length); break; }
    case 'H': { buf = reduced_min_kernel_ushort(ptr, length); break; }
    case 'i': { buf = reduced_min_kernel_int(ptr, length); break; }
    case 'I': { buf = reduced_min_kernel_uint(ptr, length); break; }
    case 'l': { buf = reduced_min_kernel_long(ptr, length); break; }
    case 'L': { buf = reduced_min_kernel_ulong(ptr, length); break; }
    case 'q': { buf = reduced_min_kernel_longlong(ptr, length); break; }
    case 'Q': { buf = reduced_min_kernel_ulonglong(ptr, length); break; }
    case 'f': { buf = reduced_min_kernel_float(ptr, length); break; }
    case 'd': { buf = reduced_min_kernel_double(ptr, length); break; }
    case 'g': { buf = reduced_min_kernel_longdouble(ptr, length); break; }
    default: {
      PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
      return NULL;
    }
  }
  if(!buf){
    PyErr_SetString(PyExc_RuntimeError, "Kernel failed");
    return NULL;
  }
  return PyCapsule_New(buf, "CPU", cpu_free);
}

static PyMethodDef methods[] = {
  {"reduce_max", reduced_max, METH_VARARGS, "get reduced max"},
  {"reduce_min", reduced_min, METH_VARARGS, "get reduced min"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cpu_ops_module = {
  PyModuleDef_HEAD_INIT,
  "cpu_ops",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cpu_ops(void){
  return PyModule_Create(&cpu_ops_module);
}
