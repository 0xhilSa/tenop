#include <python3.10/Python.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>


void cpu_free(PyObject *pycapsule){
  void *ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr) free(ptr);
}

static void *kernel_bool(void *ptr, Py_ssize_t length){
  bool *buf = malloc(sizeof(bool));
  memset(buf, 0U, sizeof(bool));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((bool *)ptr)[i] > *buf){ *buf = ((bool*)ptr)[i]; }
  }
  return buf;
}

static void *kernel_char(void *ptr, Py_ssize_t length){
  char *buf = malloc(sizeof(char));
  memset(buf, 0, sizeof(char));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((char *)ptr)[i] > *buf){ *buf = ((char *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_uchar(void *ptr, Py_ssize_t length){
  char *buf = malloc(sizeof(unsigned char));
  memset(buf, 0U, sizeof(unsigned char));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned char *)ptr)[i] > *buf){ *buf = ((unsigned char *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_short(void *ptr, Py_ssize_t length){
  short *buf = malloc(sizeof(short));
  memset(buf, 0, sizeof(short));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((short *)ptr)[i] > *buf){ *buf = ((short *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_ushort(void *ptr, Py_ssize_t length){
  unsigned short *buf = malloc(sizeof(unsigned short));
  memset(buf, 0U, sizeof(unsigned short));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned short *)ptr)[i] > *buf){ *buf = ((unsigned short *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_int(void *ptr, Py_ssize_t length){
  int *buf = malloc(sizeof(int));
  memset(buf, 0, sizeof(int));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((int *)ptr)[i] > *buf){ *buf = ((int *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_uint(void *ptr, Py_ssize_t length){
  unsigned int *buf = malloc(sizeof(unsigned int));
  memset(buf, 0U, sizeof(unsigned int));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned int *)ptr)[i] > *buf){ *buf = ((unsigned int *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_long(void *ptr, Py_ssize_t length){
  long *buf = malloc(sizeof(long));
  memset(buf, 0, sizeof(long));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long *)ptr)[i] > *buf){ *buf = ((long *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_ulong(void *ptr, Py_ssize_t length){
  unsigned long *buf = malloc(sizeof(unsigned long));
  memset(buf, 0U, sizeof(unsigned long));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned long *)ptr)[i] > *buf){ *buf = ((unsigned long *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_longlong(void *ptr, Py_ssize_t length){
  long long *buf = malloc(sizeof(long long));
  memset(buf, 0, sizeof(long long));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long long *)ptr)[i] > *buf){ *buf = ((long long *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_ulonglong(void *ptr, Py_ssize_t length){
  unsigned long long *buf = malloc(sizeof(unsigned long long));
  memset(buf, 0U, sizeof(unsigned long long));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((unsigned long long *)ptr)[i] > *buf){ *buf = ((unsigned long long *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_float(void *ptr, Py_ssize_t length){
  float *buf = malloc(sizeof(float));
  memset(buf, 0.0f, sizeof(float));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((float *)ptr)[i] > *buf){ *buf = ((float *)ptr)[i]; }
  }
  return buf;
}

static void *kernel_double(void *ptr, Py_ssize_t length){
  double *buf = malloc(sizeof(double));
  memset(buf, 0.0, sizeof(double));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((double *)ptr)[i] > *buf){ *buf = ((double *)ptr)[i]; }
  }
  return buf;
}
static void *kernel_longdouble(void *ptr, Py_ssize_t length){
  long double *buf = malloc(sizeof(long double));
  memset(buf, 0.0L, sizeof(long double));
  for(Py_ssize_t i = 0; i < length; i++){
    if(((long double *)ptr)[i] > *buf){ *buf = ((long double *)ptr)[i]; }
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
    case '?': { buf = kernel_bool(ptr, length); break; }
    case 'b': { buf = kernel_char(ptr, length); break; }
    case 'B': { buf = kernel_uchar(ptr, length); break; }
    case 'h': { buf = kernel_short(ptr, length); break; }
    case 'H': { buf = kernel_ushort(ptr, length); break; }
    case 'i': { buf = kernel_int(ptr, length); break; }
    case 'I': { buf = kernel_uint(ptr, length); break; }
    case 'l': { buf = kernel_long(ptr, length); break; }
    case 'L': { buf = kernel_ulong(ptr, length); break; }
    case 'q': { buf = kernel_longlong(ptr, length); break; }
    case 'Q': { buf = kernel_ulonglong(ptr, length); break; }
    case 'f': { buf = kernel_float(ptr, length); break; }
    case 'd': { buf = kernel_double(ptr, length); break; }
    case 'g': { buf = kernel_longdouble(ptr, length); break; }
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
