#include <python3.10/Python.h>
#include <complex.h>
#include <python3.10/complexobject.h>
#include <python3.10/iterobject.h>
#include <python3.10/listobject.h>
#include <python3.10/modsupport.h>
#include <python3.10/object.h>
#include <python3.10/pycapsule.h>
#include <python3.10/pyerrors.h>
#include <stdbool.h>

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
    case 'F': return sizeof(float _Complex);
    case 'D': return sizeof(double _Complex);
    case 'G': return sizeof(long double _Complex);  // it may cause problem since in cuda we are allocating double _Complex only
    default:  return 0;
  }
}

static void cpu_free(PyObject* pycapsule){
  void* host_ptr = PyCapsule_GetPointer(pycapsule, "CPU");
  if(host_ptr) free(host_ptr);
}

static PyObject* to_cpu(PyObject* self, PyObject* args){
  PyObject* pylist;
  const char* fmt;
  if(!PyArg_ParseTuple(args, "Os", &pylist, &fmt)) return NULL;
  if(!PyList_Check(pylist)){
    PyErr_SetString(PyExc_TypeError, "Expected a Python List");
    return NULL;
  }
  size_t length = PyList_Size(pylist);
  size_t size = sizeofs(fmt);
  if(size == 0){
    PyErr_Format(PyExc_ValueError, "Invalid Fmt: %s", fmt);
    return NULL;
  }
  size_t total_size = (size_t)length * size;
  void* buffer = malloc(total_size);
  for(Py_ssize_t i = 0; i < length; i++){
    PyObject* item = PyList_GetItem(pylist, i);
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
                  Py_complex cmpx = PyComplex_AsCComplex(item);
                  ((float _Complex*)buffer)[i] = (float)cmpx.real + (float)cmpx.imag * I;
                  break;
                }
      case 'D': {
                  Py_complex cmpx = PyComplex_AsCComplex(item);
                  ((double _Complex*)buffer)[i] = (double)cmpx.real + (double)cmpx.imag * I;
                  break;
                }
      case 'G': {
                  Py_complex cmpx = PyComplex_AsCComplex(item);
                  ((long double _Complex*)buffer)[i] = (long double)cmpx.real + (long double)cmpx.imag * I;
                  break;
                }
      default:  {
                  free(buffer);
                  PyErr_SetString(PyExc_ValueError, "Type Conversion Failed :(");
                  return NULL;
                }
    }
  }
  return PyCapsule_New(buffer, "CPU", cpu_free);
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
  void* buffer = PyCapsule_GetPointer(pycapsule, "CPU");
  if(!buffer){
    PyErr_SetString(PyExc_ValueError, "Invalid capsule pointer");
    return NULL;
  }
  PyObject* pylist = PyList_New(length);
  if(!pylist) return NULL;
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
        float _Complex c = ((float _Complex*)buffer)[i];
        item = PyComplex_FromDoubles(crealf(c), cimagf(c));
        break;
      }
      case 'D': {
        double _Complex c = ((double _Complex*)buffer)[i];
        item = PyComplex_FromDoubles(creal(c), cimag(c));
        break;
      }
      case 'G': {
        long double _Complex c = ((long double _Complex*)buffer)[i];
        item = PyComplex_FromDoubles(creall(c), cimagl(c));
        break;
      }
      default:
        Py_DECREF(pylist);
        PyErr_SetString(PyExc_ValueError, "Unsupported format in to_list()");
        return NULL;
    }
    if(!item){
      Py_DECREF(pylist);
      return NULL;
    }
    PyList_SET_ITEM(pylist, i, item);
  }
  return pylist;
}

static PyMethodDef methods[] = {
  {"toCPU", to_cpu, METH_VARARGS, "Convert list to raw CPU buffer"},
  {"toList", to_list, METH_VARARGS, "Convert CPU buffer to Python list"},
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

