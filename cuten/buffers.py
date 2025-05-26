from __future__ import annotations
from types import NoneType
from typing import Union, Type
import numpy as np
from .device import Device
from .helpers import TensorType, flatten, has_uniform_shape, infer_shape, reshape
from .dtypes import *
from .shape import Shape
from .engine import cpu


class BaseBuffer:
  def __init__(self, buffer:TensorType, dtype:Union[DType,Type,NoneType]=None, device:str="cpu:0", requires_grad:bool=False, const:bool=False):
    if not has_uniform_shape(buffer): raise ValueError("Buffer must have uniform shape")
    self.__shape = Shape(infer_shape(buffer))
    self.__ndim = len(self.__shape)
    self.__device = Device(device)
    self.__requires_grad = requires_grad
    self.__const = const
    __flatten = flatten(buffer)
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
    else: raise NotImplementedError
  @property
  def shape(self): return self.__shape
  @property
  def ndim(self): return self.__ndim
  @property
  def dtype(self): return self.__dtype
  @property
  def device(self): return self.__device
  @property
  def ptr(self): return self.__pointer
  @property
  def stride(self): return self.__stride
  def requires_grad(self): return self.__requires_grad
  def is_const(self): return self.__const
  def sizeof(self):
    length = 1
    for x in self.__shape.tolist(): length *= x
    return length * self.__dtype.nbyte
  def numpy(self):
    if self.__device.type_ == "cpu": return np.array(reshape(cpu.data(self.ptr, self.shape.totuple(), self.dtype.fmt), self.__shape.totuple()))
    elif self.__device.type_ == "cuda": raise NotImplementedError

class Buffer(BaseBuffer):
  def __repr__(self): return f"<Buffer(shape={self.shape.totuple()}, dtype='{self.dtype.ctype}', device='{self.device.type_}:{self.device.index}', requires_grad={self.requires_grad()}, const={self.is_const()})>"

class LazyBuffer(BaseBuffer):
  def __repr__(self): return f"<LazyBuffer(shape={self.shape.totuple()}, dtype='{self.dtype.ctype}', device='{self.device.type_}:{self.device.index}', requires_grad={self.requires_grad()}, const={self.is_const()})>"
