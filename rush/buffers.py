from __future__ import annotations
from typing import Union, List, Literal, Type
from .engine import cpu_core, cuda_core
from .dtypes import *

Scalars = Union[int, float, complex, bool]
Devices = Literal["cuda", "cpu"]
Backends = {"cuda": cuda_core, "cpu": cpu_core}
DTypes = INTS + UINTS + FP + CMPX + BOOL


class Buffer:
  def __init__(
    self,
    obj: Union[List[Scalars], Scalars],
    dtype: DType = float64,
    device: Devices = "cpu",
    const: bool = False
  ):
    self.__pointer, self.__length, self.__kind = self._check_input(obj, dtype, device)
    self.__dtype = dtype
    self.__nbytes = self.__length * self.__dtype.size
    self.__nbits = self.__dtype.size * 8
    self.__device = device
    self.__const = const
  def __repr__(self): return f"<Buffer(length={self.__length}, dtype='{self.__dtype.ctype}', device={self.__device.upper()}, bytes={self.__nbytes}, const={self.__const})>"
  @property
  def pointer(self): return self.__pointer
  @property
  def length(self): return self.__length
  @property
  def dtype(self): return self.__dtype
  @property
  def device(self): return self.__device
  @property
  def backend(self): return Backends[self.__device]
  @property
  def fmt(self): return self.__dtype.fmt
  @property
  def ctype(self): return self.__dtype.ctype
  @property
  def nbytes(self): return self.__nbytes
  @property
  def nbits(self): return self.__nbits
  @staticmethod
  def _from_custom2builtin(dtype: DType) -> Type:
    if dtype in INTS + UINTS: return int
    if dtype in FP: return float
    if dtype in CMPX: return complex
    return bool
  @staticmethod
  def _from_builtin2custom(dtype: Type) -> DType:
    if dtype == int: return int64
    if dtype == float: return float64
    if dtype == complex: return complex128
    return boolean
  @staticmethod
  def _check_input(
    obj: Union[List[Scalars], Scalars],
    dtype: DType,
    device: Devices
  ) -> tuple:
    if isinstance(obj, list):
      if any(isinstance(item, list) for item in obj): raise TypeError("Only 1D arrays are supported")
      py_type = Buffer._from_custom2builtin(dtype)
      obj = [py_type(x) for x in obj]
      length = len(obj)
      dispatch = {
        "cpu": lambda: cpu_core.v_toCPU(obj, dtype.fmt),
        "cuda": lambda: cuda_core.v_toCUDA(obj, dtype.fmt)
      }
      kind = "v"
    elif isinstance(obj, (int, float, complex, bool)):
      py_type = Buffer._from_custom2builtin(dtype) if not isinstance(dtype, type) else dtype
      obj = py_type(obj)
      dtype = Buffer._from_builtin2custom(py_type)
      length = 1
      dispatch = {
        "cpu": lambda: cpu_core.s_toCPU(obj, dtype.fmt),
        "cuda": lambda: cuda_core.s_toCUDA(obj, dtype.fmt)
      }
      kind = "s"
    else: raise TypeError(f"Unsupported object type: {type(obj).__name__}")
    if device not in dispatch: raise ValueError(f"Unsupported device: '{device}'")
    pointer = dispatch[device]()
    return pointer, length, kind

  def __getitem__(self, index:Union[int, slice]): raise NotImplementedError
  def __setitem__(self, index, value:Union[int, slice]): raise NotImplementedError

  def cuda(self):
    if self.__device == "cpu":
      if self.__kind == "s":
        scalar = cpu_core.s_fromCPU(self.__pointer, self.__dtype.fmt)
        return Buffer(scalar, dtype=self.__dtype, device="cuda")
      elif self.__kind == "v":
        vector = cpu_core.v_fromCPU(self.__pointer, self.__length, self.__dtype.fmt)
        return Buffer(vector, dtype=self.__dtype, device="cuda")
    return self
  def cpu(self):
    if self.__device == "cpu":
      if self.__kind == "s":
        scalar = cuda_core.s_fromCUDA(self.__pointer, self.__dtype.fmt)
        return Buffer(scalar, dtype=self.__dtype, device="cpu")
      elif self.__kind == "v":
        vector = cuda_core.v_fromCUDA(self.__pointer, self.__length, self.__dtype.fmt)
        return Buffer(vector, dtype=self.__dtype, device="cpu")
    return self
  def isvector(self): return self.__kind == "v"
  def isscalar(self): return self.__kind == "s"

  def append(self): raise NotImplementedError
  def remove(self): raise NotImplementedError
