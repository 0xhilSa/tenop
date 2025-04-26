#include <python3.10/Python.h>
#include <complex.h>
#include <stdbool.h>

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

static void cpu_free(PyObject* pycapsule){
  void* ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr) free(ptr);
}

static PyObject* s_tocpu(PyObject* self, PyObject* args){
  PyObject* pyscalar;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pyscalar, &fmt)) return NULL;
  size_t size = get_size(fmt);
  if(size == (size_t)-1){
    PyErr_SetString(PyExc_ValueError, "Fmt can't be NULL");
    return NULL;
  }else if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  void* ptr = malloc(size);
  switch(*fmt){
    case '?': *(bool*)ptr = PyObject_IsTrue(pyscalar); break;
    case 'b': *(char*)ptr = (char)PyLong_AsLong(pyscalar); break;
    case 'B': *(unsigned char*)ptr = (unsigned char)PyLong_AsUnsignedLong(pyscalar); break;
    case 'h': *(short*)ptr = (short)PyLong_AsLong(pyscalar); break;
    case 'H': *(unsigned short*)ptr = (unsigned short)PyLong_AsUnsignedLong(pyscalar); break;
    case 'i': *(int*)ptr = (int)PyLong_AsLong(pyscalar); break;
    case 'I': *(unsigned int*)ptr = (unsigned int)PyLong_AsUnsignedLong(pyscalar); break;
    case 'l': *(long*)ptr = PyLong_AsLong(pyscalar); break;
    case 'L': *(unsigned long*)ptr = PyLong_AsUnsignedLong(pyscalar); break;
    case 'q': *(long long*)ptr = PyLong_AsLongLong(pyscalar); break;
    case 'Q': *(unsigned long long*)ptr = PyLong_AsUnsignedLongLong(pyscalar); break;
    case 'f': *(float*)ptr = (float)PyFloat_AsDouble(pyscalar); break;
    case 'd': *(double*)ptr = PyFloat_AsDouble(pyscalar); break;
    case 'g': *(long double*)ptr = (long double)PyFloat_AsDouble(pyscalar); break;
    case 'F': *(float _Complex*)ptr = (float _Complex)PyComplex_RealAsDouble(pyscalar) + I * (float _Complex)PyComplex_ImagAsDouble(pyscalar); break;
    case 'D': *(double _Complex*)ptr = PyComplex_RealAsDouble(pyscalar) + I * PyComplex_ImagAsDouble(pyscalar); break;
    case 'G': *(long double _Complex*)ptr = (long double)PyComplex_RealAsDouble(pyscalar) + I * (long double)PyComplex_ImagAsDouble(pyscalar); break;
    default:
      free(ptr);
      PyErr_Format(PyExc_ValueError, "Unsupported format character: %c", *fmt);
      return NULL;
  }
  if(PyErr_Occurred()){
    free(ptr);
    return NULL;
  }
  return PyCapsule_New(ptr, "CPU", cpu_free);
}

static PyObject* s_fromcpu(PyObject* self, PyObject* args){
  PyObject* pycapsule;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pycapsule, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "Expected a PyCapsule/Pointer");
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
  void* ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid or NULL capsule pointer");
    return NULL;
  }
  switch(*fmt){
    case '?': return PyBool_FromLong(*(bool*)ptr);
    case 'b': return PyLong_FromLong(*(char*)ptr);
    case 'B': return PyLong_FromUnsignedLong(*(unsigned char*)ptr);
    case 'h': return PyLong_FromLong(*(short*)ptr);
    case 'H': return PyLong_FromUnsignedLong(*(unsigned short*)ptr);
    case 'i': return PyLong_FromLong(*(int*)ptr);
    case 'I': return PyLong_FromUnsignedLong(*(unsigned int*)ptr);
    case 'l': return PyLong_FromLong(*(long*)ptr);
    case 'L': return PyLong_FromUnsignedLong(*(unsigned long*)ptr);
    case 'q': return PyLong_FromLongLong(*(long long*)ptr);
    case 'Q': return PyLong_FromUnsignedLongLong(*(unsigned long long*)ptr);
    case 'f': return PyFloat_FromDouble(*(float*)ptr);
    case 'd': return PyFloat_FromDouble(*(double*)ptr);
    case 'g': return PyFloat_FromDouble((double)*(long double*)ptr);
    case 'F': {
      float _Complex c = *(float _Complex*)ptr;
      return PyComplex_FromDoubles(crealf(c), cimagf(c));
    }
    case 'D': {
      double _Complex c = *(double _Complex*)ptr;
      return PyComplex_FromDoubles(creal(c), cimag(c));
    }
    case 'G': {
      long double _Complex c = *(long double _Complex*)ptr;
      return PyComplex_FromDoubles((double)creall(c), (double)cimagl(c));
    }
    default:
      PyErr_Format(PyExc_ValueError, "Unsupported format character: %c", *fmt);
      return NULL;
  }
}

static PyObject* v_tocpu(PyObject* self, PyObject* args){
  PyObject* pylist;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pylist, &fmt)) return NULL;
  if(!PyList_Check(pylist)){
    PyErr_SetString(PyExc_TypeError, "Expected a Python List");
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
  void* ptr = malloc((size_t)length * size);
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject* item = PyList_GetItem(pylist, i);
    switch(*fmt){
      case '?': ((bool*)ptr)[i] = (bool)PyObject_IsTrue(item); break;
      case 'b': ((char*)ptr)[i] = (char)PyLong_AsLong(item); break;
      case 'B': ((unsigned char*)ptr)[i] = (unsigned char)PyLong_AsUnsignedLong(item); break;
      case 'h': ((short*)ptr)[i] = (short)PyLong_AsLong(item); break;
      case 'H': ((unsigned short*)ptr)[i] = (unsigned short)PyLong_AsUnsignedLong(item); break;
      case 'i': ((int*)ptr)[i] = (int)PyLong_AsLong(item); break;
      case 'I': ((unsigned int*)ptr)[i] = (unsigned int)PyLong_AsUnsignedLong(item); break;
      case 'l': ((long*)ptr)[i] = (long)PyLong_AsLong(item); break;
      case 'L': ((unsigned long*)ptr)[i] = (unsigned long)PyLong_AsUnsignedLong(item); break;
      case 'q': ((long long*)ptr)[i] = (long long)PyLong_AsLongLong(item); break;
      case 'Q': ((unsigned long long*)ptr)[i] = (unsigned long long)PyLong_AsUnsignedLongLong(item); break;
      case 'f': ((float*)ptr)[i] = (float)PyFloat_AsDouble(item); break;
      case 'd': ((double*)ptr)[i] = (double)PyFloat_AsDouble(item); break;
      case 'g': ((long double*)ptr)[i] = (long double)PyFloat_AsDouble(item); break;
      case 'F': {
                  float real = (float)PyComplex_RealAsDouble(item);
                  float imag = (float)PyComplex_ImagAsDouble(item);
                  ((float _Complex*)ptr)[i] = real + imag * I;
                  break;
                }
      case 'D': {
                  double real = (double)PyComplex_RealAsDouble(item);
                  double imag = (double)PyComplex_ImagAsDouble(item);
                  ((double _Complex*)ptr)[i] = real + imag * I;
                  break;
                }
      case 'G': {
                  long double real = (long double)PyComplex_RealAsDouble(item);
                  long double imag = (long double)PyComplex_ImagAsDouble(item);
                  ((long double _Complex*)ptr)[i] = real + imag * I;
                  break;
                }
      default:  {
                  free(ptr);
                  PyErr_Format(PyExc_ValueError, "Unsupported format character: %c", *fmt);
                  return NULL;
                }
    }
    if(PyErr_Occurred()){
      free(ptr);
      return NULL;
    }
  }
  return PyCapsule_New(ptr, "CPU", cpu_free);
}

static PyObject* v_fromcpu(PyObject* slef, PyObject* args){
  PyObject* pycapsule;
  Py_ssize_t length;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Ons", &pycapsule, &length, &fmt)) return NULL;
  if(!PyCapsule_CheckExact(pycapsule)){
    PyErr_SetString(PyExc_ValueError, "Expected a PyCapsule/Pointer");
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
  void* ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(ptr == NULL){
    PyErr_SetString(PyExc_ValueError, "Invalid or NULL capsule pointer");
    return NULL;
  }
  PyObject* pylist = PyList_New(length);
  if(!pylist) return NULL;
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject* item = NULL;
    switch(*fmt){
      case '?': item = PyBool_FromLong(((bool*)ptr)[i]); break;
      case 'b': item = PyLong_FromLong(((char*)ptr)[i]); break;
      case 'B': item = PyLong_FromUnsignedLong(((unsigned char*)ptr)[i]); break;
      case 'h': item = PyLong_FromLong(((short*)ptr)[i]); break;
      case 'H': item = PyLong_FromUnsignedLong(((unsigned short*)ptr)[i]); break;
      case 'i': item = PyLong_FromLong(((int*)ptr)[i]); break;
      case 'I': item = PyLong_FromUnsignedLong(((unsigned int*)ptr)[i]); break;
      case 'l': item = PyLong_FromLong(((long*)ptr)[i]); break;
      case 'L': item = PyLong_FromUnsignedLong(((unsigned long*)ptr)[i]); break;
      case 'q': item = PyLong_FromLongLong(((long long*)ptr)[i]); break;
      case 'Q': item = PyLong_FromUnsignedLongLong(((unsigned long long*)ptr)[i]); break;
      case 'f': item = PyFloat_FromDouble(((float*)ptr)[i]); break;
      case 'd': item = PyFloat_FromDouble(((double*)ptr)[i]); break;
      case 'g': item = PyFloat_FromDouble(((long double*)ptr)[i]); break;
      case 'F': {
        float _Complex val = ((float _Complex*)ptr)[i];
        item = PyComplex_FromDoubles(crealf(val), cimagf(val));
        break;
      }
      case 'D': {
        double _Complex val = ((double _Complex*)ptr)[i];
        item = PyComplex_FromDoubles(crealf(val), cimagf(val));
        break;
      }
      case 'G': {
        long double _Complex val = ((long double _Complex*)ptr)[i];
        item = PyComplex_FromDoubles(crealf(val), cimagf(val));
        break;
      }
      default:  {
        Py_DECREF(pylist);
        PyErr_Format(PyExc_ValueError, "Unsupported format character: %c", *fmt);
        return NULL;
      }
    }
    if(PyErr_Occurred()){
      free(ptr);
      return NULL;
    }
    PyList_SET_ITEM(pylist, i, item);
  }
  return pylist;
}

static PyMethodDef methods[] = {
  {"s_toCPU", s_tocpu, METH_VARARGS, "Convert scalar to raw CPU buffer"},
  {"s_fromCPU", s_fromcpu, METH_VARARGS, "from CPU buffer to scalar"},
  {"v_toCPU", v_tocpu, METH_VARARGS, "Convert vector to raw CPU buffer"},
  {"v_fromCPU", v_fromcpu, METH_VARARGS, "from CPU buffer to vector"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "cpu_core",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC PyInit_cpu_core(void){
  return PyModule_Create(&module);
}
