#include <python3.10/Python.h>
#include <stdbool.h>
#include <complex.h>


static inline void *fail_exit(void *buffer){
  free(buffer);
  return NULL;
}

void cpu_free(PyObject *pycapsule){
  void *ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr) free(ptr);
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
    case 'd': return sizeof(double);
    case 'g': return sizeof(long double);
    case 'F': return sizeof(float _Complex);
    case 'D': return sizeof(double _Complex);
    case 'G': return sizeof(long double _Complex);
    default: return 0;
  }
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
      case 'd': {
        double val = (double)PyFloat_AsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'g': {
        long double val = (long double)PyFloat_AsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        memcpy(dest, &val, size);
        break;
      }
      case 'F': {
        float real = (float)PyComplex_RealAsDouble(item);
        float imag = (float)PyComplex_ImagAsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        float _Complex val = real + imag * I;
        memcpy(dest, &val, size);
        break;
      }
      case 'D': {
        double real = (double)PyComplex_RealAsDouble(item);
        double imag = (double)PyComplex_ImagAsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        double _Complex val = real + imag * I;
        memcpy(dest, &val, size);
        break;
      }
      case 'G': {
        long double real = (long double)PyComplex_RealAsDouble(item);
        long double imag = (long double)PyComplex_ImagAsDouble(item);
        if(PyErr_Occurred()) return fail_exit(buffer);
        long double _Complex val = real + imag * I;
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

void *__typecasting_scalar(PyObject *scalar, const char *fmt){
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return 0;
  }
  void *buffer = malloc(size);
  switch(*fmt){
    case '?': {
      bool val = (bool)PyObject_IsTrue(scalar);
      memcpy(buffer, &val, size);
      break;
    }
    case 'b': {
      char val = (char)PyLong_AsLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'B': {
      unsigned char val = (unsigned char)PyLong_AsUnsignedLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'h': {
      short val = (short)PyLong_AsLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'H': {
      unsigned short val = (unsigned short)PyLong_AsUnsignedLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'i': {
      int val = (int)PyLong_AsLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'I': {
      unsigned int val = (unsigned int)PyLong_AsUnsignedLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'l': {
      long val = (long)PyLong_AsLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'L': {
      unsigned long val = (unsigned long)PyLong_AsUnsignedLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'q': {
      long long val = (long long)PyLong_AsLongLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'Q': {
      unsigned long long val = (unsigned long long)PyLong_AsUnsignedLongLong(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'f': {
      float val = (float)PyFloat_AsDouble(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'd': {
      double val = (double)PyFloat_AsDouble(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'g': {
      long double val = (long double)PyFloat_AsDouble(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      memcpy(buffer, &val, size);
      break;
    }
    case 'F': {
      float real = (float)PyComplex_RealAsDouble(scalar);
      float imag = (float)PyComplex_ImagAsDouble(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      float _Complex val = real + imag * I;
      memcpy(buffer, &val, size);
      break;
    }
    case 'D': {
      double real = (double)PyComplex_RealAsDouble(scalar);
      double imag = (double)PyComplex_ImagAsDouble(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      double _Complex val = real + imag * I;
      memcpy(buffer, &val, size);
      break;
    }
    case 'G': {
      long double real = (long double)PyComplex_RealAsDouble(scalar);
      long double imag = (long double)PyComplex_ImagAsDouble(scalar);
      if(PyErr_Occurred()) return fail_exit(buffer);
      long double _Complex val = real + imag * I;
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

static PyObject *tohost(PyObject *self, PyObject *args){
  PyObject *pylist;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "Os", &pylist, &fmt)) return NULL;
  void *buffer;
  if(PyList_Check(pylist)){
    Py_ssize_t length = PyList_Size(pylist);
    buffer = __typecasting_list(pylist, length, fmt);
  }else{
    buffer = __typecasting_scalar(pylist, fmt);
  }
  return PyCapsule_New(buffer, "CPU", cpu_free);
}

static PyObject *__scalar(void *buffer, const char *fmt){
  switch(*fmt){
    case '?': return PyBool_FromLong(*(bool*)buffer);
    case 'b': return PyLong_FromLong(*(char*)buffer);
    case 'B': return PyLong_FromUnsignedLong(*(unsigned char*)buffer);
    case 'h': return PyLong_FromLong(*(short*)buffer);
    case 'H': return PyLong_FromUnsignedLong(*(unsigned short*)buffer);
    case 'i': return PyLong_FromLong(*(int*)buffer);
    case 'I': return PyLong_FromUnsignedLong(*(unsigned int*)buffer);
    case 'l': return PyLong_FromLong(*(long*)buffer);
    case 'L': return PyLong_FromUnsignedLong(*(unsigned long*)buffer);
    case 'q': return PyLong_FromLongLong(*(long long*)buffer);
    case 'Q': return PyLong_FromUnsignedLongLong(*(unsigned long long*)buffer);
    case 'f': return PyFloat_FromDouble(*(float*)buffer);
    case 'd': return PyFloat_FromDouble(*(double*)buffer);
    case 'g': return PyFloat_FromDouble(*(long double*)buffer);
    case 'F': {
      float *val = (float*)buffer;
      return PyComplex_FromDoubles((double)val[0], (double)val[1]);
    }
    case 'D': {
      double *val = (double*)buffer;
      return PyComplex_FromDoubles((double)val[0], (double)val[1]);
    }
    case 'G': {
      long double *val = (long double*)buffer;
      return PyComplex_FromDoubles((long double)val[0], (long double)val[1]);
    }
  }
  PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
  return NULL;
}

static PyObject *__list(void *buffer, Py_ssize_t length, const char *fmt){
  PyObject *pylist = PyList_New(length);
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject *item = NULL;
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
        float _Complex val = ((float _Complex*)buffer)[i];
        item = PyComplex_FromDoubles(crealf(val), cimagf(val));
        break;
      }
      case 'D': {
        double _Complex val = ((double _Complex*)buffer)[i];
        item = PyComplex_FromDoubles(creall(val), (cimagl(val)));
        break;
      }
      case 'G': {
        long double _Complex val = ((long double _Complex*)buffer)[i];
        item = PyComplex_FromDoubles(creall(val), cimagl(val));
        break;
      }
    }
    if(!item){
      Py_DECREF(pylist);
      return NULL;
    }
    PyList_SET_ITEM(pylist, i, item);
  }
  return pylist;
}

static PyObject *todata(PyObject *self, PyObject *args){
  PyObject *pycapsule, *shape;
  const char *fmt;
  if(!PyArg_ParseTuple(args, "OOs", &pycapsule, &shape, &fmt)) return NULL;
  void *buffer = PyCapsule_GetPointer(pycapsule, "CPU");
  if(!buffer) return NULL;
  size_t size = get_size(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid DType: %s", fmt);
    return NULL;
  }
  if(PyTuple_Size(shape) == 0){
    return __scalar(buffer, fmt);
  }
  Py_ssize_t shape_length = PyTuple_Size(shape);
  Py_ssize_t length = 1;
  for(Py_ssize_t i = 0; i < shape_length; i++){
    PyObject *item = PyTuple_GetItem(shape, i);
    length *= (size_t)PyLong_AsLong(item);
  }
  return __list(buffer, length, fmt);
}

static PyObject* get_device_name(PyObject *self, PyObject *args){
  FILE *fp = fopen("/proc/cpuinfo", "r");
  if(!fp) return PyErr_Format(PyExc_OSError, "Failed to open /proc/cpuinfo");
  char line[256];
  while(fgets(line, sizeof(line), fp)){
    if(strncmp(line, "model name", 10) == 0){
      fclose(fp);
      char *model = strchr(line, ':');
      if(model){
        while(*model && isspace(*model)) model++;
        char *newline = strchr(model, '\n');
        if(newline) *newline = '\0';
        return PyUnicode_FromString(model);
      }
    }
  }
  fclose(fp);
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
  {"tohost", tohost, METH_VARARGS, "store tensor"},
  {"data", todata, METH_VARARGS, "to list"},
  {"get_device_name", get_device_name, METH_NOARGS, "get CPU device name"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cpu",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cpu(void){
  return PyModule_Create(&module);
}
