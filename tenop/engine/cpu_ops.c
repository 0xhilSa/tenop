#include <limits.h>
#include <float.h>
#include <python3.10/Python.h>
#include <complex.h>
#include <python3.10/methodobject.h>
#include <stdbool.h>


void cpu_free(PyObject *pycapsule){
  void *ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr) free(ptr);
}

static PyObject *get_reduced_max_bool(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  bool *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  bool *result = malloc(sizeof(bool));
  if(result == NULL) return PyErr_NoMemory();
  *result = 0U;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_char(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  char *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  char *result = malloc(sizeof(char));
  if(result == NULL) return PyErr_NoMemory();
  *result = SCHAR_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_uchar(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned char *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned char *result = malloc(sizeof(unsigned char));
  if(result == NULL) return PyErr_NoMemory();
  *result = 0U;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_short(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  short *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  short *result = malloc(sizeof(short));
  if(result == NULL) return PyErr_NoMemory();
  *result = SHRT_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_ushort(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned short *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned short *result = malloc(sizeof(unsigned short));
  if(result == NULL) return PyErr_NoMemory();
  *result = 0U;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_int(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  int *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  int *result = malloc(sizeof(int));
  if(result == NULL) return PyErr_NoMemory();
  *result = INT_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_uint(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned int *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned int *result = malloc(sizeof(unsigned int));
  if(result == NULL) return PyErr_NoMemory();
  *result = 0U;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_long(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  long *result = malloc(sizeof(long));
  if(result == NULL) return PyErr_NoMemory();
  *result = LONG_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_ulong(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned long *result = malloc(sizeof(unsigned long));
  if(result == NULL) return PyErr_NoMemory();
  *result = 0U;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_longlong(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  long long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  long long *result = malloc(sizeof(long long));
  if(result == NULL) return PyErr_NoMemory();
  *result = LLONG_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_ulonglong(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned long long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned long long *result = malloc(sizeof(unsigned long long));
  if(result == NULL) return PyErr_NoMemory();
  *result = 0U;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_float(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  float *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  float *result = malloc(sizeof(float));
  if(result == NULL) return PyErr_NoMemory();
  *result = FLT_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_double(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  double *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  double *result = malloc(sizeof(double));
  if(result == NULL) return PyErr_NoMemory();
  *result = DBL_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_max_longdouble(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  long double *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  long double *result = malloc(sizeof(long double));
  if(result == NULL) return PyErr_NoMemory();
  *result = LDBL_MIN;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] > *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_bool(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  bool *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  bool *result = malloc(sizeof(bool));
  if(result == NULL) return PyErr_NoMemory();
  *result = UCHAR_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_char(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  char *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  char *result = malloc(sizeof(char));
  if(result == NULL) return PyErr_NoMemory();
  *result = CHAR_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_uchar(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned char *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned char *result = malloc(sizeof(unsigned char));
  if(result == NULL) return PyErr_NoMemory();
  *result = UCHAR_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_short(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  short *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  short *result = malloc(sizeof(short));
  if(result == NULL) return PyErr_NoMemory();
  *result = SHRT_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_ushort(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned short *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned short *result = malloc(sizeof(unsigned short));
  if(result == NULL) return PyErr_NoMemory();
  *result = USHRT_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_int(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  int *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  int *result = malloc(sizeof(int));
  if(result == NULL) return PyErr_NoMemory();
  *result = INT_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_uint(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned int *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned int *result = malloc(sizeof(unsigned int));
  if(result == NULL) return PyErr_NoMemory();
  *result = UINT_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_long(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  long *result = malloc(sizeof(long));
  if(result == NULL) return PyErr_NoMemory();
  *result = LONG_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_ulong(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned long *result = malloc(sizeof(unsigned long));
  if(result == NULL) return PyErr_NoMemory();
  *result = ULONG_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_longlong(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  long long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  long long *result = malloc(sizeof(long long));
  if(result == NULL) return PyErr_NoMemory();
  *result = LLONG_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_ulonglong(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  unsigned long long *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  unsigned long long *result = malloc(sizeof(unsigned long long));
  if(result == NULL) return PyErr_NoMemory();
  *result = ULLONG_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_float(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  float *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  float *result = malloc(sizeof(float));
  if(result == NULL) return PyErr_NoMemory();
  *result = FLT_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_double(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  double *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  double *result = malloc(sizeof(double));
  if(result == NULL) return PyErr_NoMemory();
  *result = DBL_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyObject *get_reduced_min_longdouble(PyObject *self, PyObject *args){
  PyObject *tensor;
  Py_ssize_t length;
  if(!PyArg_ParseTuple(args, "On", &tensor, &length)) return NULL;
  long double *buffer = PyCapsule_GetPointer(tensor, "CPU");
  if(buffer == NULL) return NULL;
  long double *result = malloc(sizeof(long double));
  if(result == NULL) return PyErr_NoMemory();
  *result = LDBL_MAX;
  for(Py_ssize_t i = 0; i < length; i++){
    if(buffer[i] < *result){
      *result = buffer[i];
    }
  }
  return PyCapsule_New(result, "CPU", cpu_free);
}

static PyMethodDef methods[] = {
  {"reduce_max_bool", get_reduced_max_bool, METH_VARARGS, "get max value for bool dtype"},
  {"reduce_max_char", get_reduced_max_char, METH_VARARGS, "get max value for char dtype"},
  {"reduce_max_uchar", get_reduced_max_uchar, METH_VARARGS, "get max value for unsigned char dtype"},
  {"reduce_max_short", get_reduced_max_short, METH_VARARGS, "get max value for short dtype"},
  {"reduce_max_ushort", get_reduced_max_ushort, METH_VARARGS, "get max value for unsigned short dtype"},
  {"reduce_max_int", get_reduced_max_int, METH_VARARGS, "get max value for int dtype"},
  {"reduce_max_uint", get_reduced_max_uint, METH_VARARGS, "get max value for unsigned int dtype"},
  {"reduce_max_long", get_reduced_max_long, METH_VARARGS, "get max value for long dtype"},
  {"reduce_max_ulong", get_reduced_max_ulong, METH_VARARGS, "get max value for unsigned long dtype"},
  {"reduce_max_longlong", get_reduced_max_longlong, METH_VARARGS, "get max value for long long dtype"},
  {"reduce_max_ulonglong", get_reduced_max_ulonglong, METH_VARARGS, "get max value for unsigned long long dtype"},
  {"reduce_max_float", get_reduced_max_float, METH_VARARGS, "get max value for float dtype"},
  {"reduce_max_double", get_reduced_max_double, METH_VARARGS, "get max value for double dtype"},
  {"reduce_max_longdouble", get_reduced_max_longdouble, METH_VARARGS, "get max value for long double dtype"},
  {"reduce_min_bool", get_reduced_min_bool, METH_VARARGS, "get min value for bool dtype"},
  {"reduce_min_char", get_reduced_min_char, METH_VARARGS, "get min value for char dtype"},
  {"reduce_min_uchar", get_reduced_min_uchar, METH_VARARGS, "get min value for unsigned char dtype"},
  {"reduce_min_short", get_reduced_min_short, METH_VARARGS, "get min value for short dtype"},
  {"reduce_min_ushort", get_reduced_min_ushort, METH_VARARGS, "get min value for unsigned short dtype"},
  {"reduce_min_int", get_reduced_min_int, METH_VARARGS, "get min value for int dtype"},
  {"reduce_min_uint", get_reduced_min_uint, METH_VARARGS, "get min value for unsigned int dtype"},
  {"reduce_min_long", get_reduced_min_long, METH_VARARGS, "get min value for long dtype"},
  {"reduce_min_ulong", get_reduced_min_ulong, METH_VARARGS, "get min value for unsigned long dtype"},
  {"reduce_min_longlong", get_reduced_min_longlong, METH_VARARGS, "get min value for long long dtype"},
  {"reduce_min_ulonglong", get_reduced_min_ulonglong, METH_VARARGS, "get min value for unsigned long long dtype"},
  {"reduce_min_float", get_reduced_min_float, METH_VARARGS, "get min value for float dtype"},
  {"reduce_min_double", get_reduced_min_double, METH_VARARGS, "get min value for double dtype"},
  {"reduce_min_longdouble", get_reduced_min_longdouble, METH_VARARGS, "get min value for long double dtype"},
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
