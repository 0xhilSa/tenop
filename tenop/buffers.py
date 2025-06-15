from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union
from .helpers import Scalar, TensorType, flatten, has_uniform_shape, infer_shape, reshape
from .engine import cpu, cuda, cpu_ops
from .shape import Shape
from .device import Device
from .dtypes import *


class Buffer:
  def __init__(self, buf:Union[Scalar,TensorType], dtype:Optional[Union[DType,Type]]=None, device:str="cpu:0", requires_grad:bool=False, const:bool=False):
    if not has_uniform_shape(buf): raise ValueError("Buffer must have uniform shape")
    self.__shape = Shape(infer_shape(buf))
    self.__device = Device(device)
    self.__requires_grad = requires_grad
    self.__const = const
    __flatten = flatten(buf)
    self.__ndim = len(self.__shape)
    __stride = []
    acc = 1
    for dim in reversed(self.__shape.tolist()):
      __stride.insert(0, acc)
      acc *= dim
    self.__stride = tuple(__stride)
    if dtype is None:
      if any(isinstance(x, complex) for x in __flatten): self.__dtype = maps[complex]
      elif any(isinstance(x, float) for x in __flatten): self.__dtype = maps[float]
      elif any(isinstance(x, int) for x in __flatten): self.__dtype = maps[int]
      elif all(isinstance(x, bool) for x in __flatten): self.__dtype = maps[bool]
      else: raise ValueError(f"Invalid DType: expected from int, float, complex, or bool")
    elif isinstance(dtype, DType): self.__dtype = dtype
    elif dtype in [int, float, complex, bool]: self.__dtype = maps[dtype]
    else: raise ValueError(f"Invalid DType: expected from int, float, complex, or bool")
    if self.__device.type_ == "cpu": self.__pointer = cpu.tohost(__flatten, self.__dtype.fmt)
    elif self.__device.type_ == "cuda": self.__pointer = cuda.tocuda(__flatten, self.__device.index, self.__shape.numel(), self.__dtype.fmt)
  def __repr__(self): return f"<Buffer(shape={self.__shape.totuple()}, dtype='{self.__dtype.ctype}', device='{str(self.__device)}', requires_grad={self.__requires_grad}, const={self.__const})>"
  def ptr(self): return self.__pointer
  @property
  def shape(self): return self.__shape
  @property
  def ndim(self): return self.__ndim
  @property
  def dtype(self): return self.__dtype
  @property
  def device(self): return self.__device
  @property
  def stride(self): return self.__stride
  @property
  def requires_grad(self): return self.__requires_grad
  @property
  def isconst(self): return self.__const
  def numel(self): return self.__shape.numel()
  def data(self):
    if self.__device.type_ == "cpu": return reshape(cpu.data(self.ptr(), self.shape.totuple(), self.dtype.fmt), self.shape.totuple())
    return reshape(cpu.data(cuda.tocpu(self.ptr(), self.numel(), self.dtype.fmt), self.shape.totuple(), self.dtype.fmt), self.shape.totuple())
  def numpy(self): return np.array(self.data())
  def sizeof(self):
    length = 1
    for x in self.__shape.tolist(): length *= x
    return length * self.__dtype.nbyte
  def __getitem__(self, index:Union[int,slice]):
    raw = self.data()
    if isinstance(raw, (int, float, complex, bool)): raise IndexError(f"invalid index for 0-dim tensor")
    if isinstance(index, int): return raw[index]
    elif isinstance(index, slice):
      if index.step < 0: raise ValueError("Step index must be a 0 or positive integer")
      return raw[index]
  def astype(self): pass
  @staticmethod
  def full(value, shape:Tuple[int,...]):
    length = Shape(shape).numel()
    return [value for _ in range(length)]
  @staticmethod
  def ones(shape:Tuple[int, ...]): return Buffer.full(value=1, shape=shape)
  @staticmethod
  def zeros(shape:Tuple[int, ...]): return Buffer.full(value=0, shape=shape)
  def reduce_max(self):
    if self.__device.type_ == "cpu":
      if self.dtype.fmt == "?": return cpu.data(cpu_ops.reduce_max_bool(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "b": return cpu.data(cpu_ops.reduce_max_char(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "B": return cpu.data(cpu_ops.reduce_max_uchar(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "h": return cpu.data(cpu_ops.reduce_max_short(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "H": return cpu.data(cpu_ops.reduce_max_ushort(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "i": return cpu.data(cpu_ops.reduce_max_int(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "I": return cpu.data(cpu_ops.reduce_max_uint(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "l": return cpu.data(cpu_ops.reduce_max_long(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "L": return cpu.data(cpu_ops.reduce_max_ulong(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "q": return cpu.data(cpu_ops.reduce_max_longlong(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "Q": return cpu.data(cpu_ops.reduce_max_ulonglong(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "f": return cpu.data(cpu_ops.reduce_max_float(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "d": return cpu.data(cpu_ops.reduce_max_double(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "g": return cpu.data(cpu_ops.reduce_max_double(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt in ["F", "D", "G"]: raise ValueError("'>' unsupported for complex dtype")
      else: raise NotImplementedError
    else: raise NotImplementedError("CUDA ops not implemented yet")
  def reduce_min(self):
    if self.__device.type_ == "cpu":
      if self.dtype.fmt == "?": return cpu.data(cpu_ops.reduce_min_bool(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "b": return cpu.data(cpu_ops.reduce_min_char(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "B": return cpu.data(cpu_ops.reduce_min_uchar(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "h": return cpu.data(cpu_ops.reduce_min_short(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "H": return cpu.data(cpu_ops.reduce_min_ushort(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "i": return cpu.data(cpu_ops.reduce_min_int(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "I": return cpu.data(cpu_ops.reduce_min_uint(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "l": return cpu.data(cpu_ops.reduce_min_long(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "L": return cpu.data(cpu_ops.reduce_min_ulong(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "q": return cpu.data(cpu_ops.reduce_min_longlong(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "Q": return cpu.data(cpu_ops.reduce_min_ulonglong(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "f": return cpu.data(cpu_ops.reduce_min_float(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "d": return cpu.data(cpu_ops.reduce_min_double(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt == "g": return cpu.data(cpu_ops.reduce_min_double(self.ptr(), self.numel()), (), self.dtype.fmt)
      elif self.dtype.fmt in ["F", "D", "G"]: raise ValueError("'>' unsupported for complex dtype")
      else: raise NotImplementedError
    else: raise NotImplementedError("CUDA ops not implemented yet")
  def max(self, other): pass


__all__ = ["Buffer"]
