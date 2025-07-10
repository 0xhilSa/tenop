from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Union
from tenop.helpers import Scalar, TensorType, flatten, has_uniform_shape, infer_shape, reshape
from tenop.engine import cpu, cuda, cpu_ops
from tenop.shape import Shape
from tenop.device import Device
from tenop.dtypes import *


class Buffer:
  def __init__(self, buf:Union[Scalar,TensorType], dtype:Optional[Union[DType,Type]]=None, device:str="cpu:0", requires_grad:bool=False, const:bool=False):
    if not has_uniform_shape(buf): raise ValueError("Buffer must have uniform shape")
    self.__shape = Shape(infer_shape(buf))
    self.__device = Device(device)
    self.__requires_grad = requires_grad
    self.__const = const
    self.__grad = None
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
  @property
  def grad(self): return self.__grad
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
  def reduce_max(self):
    if self.__device.type_ == "cpu": return cpu.data(cpu_ops.reduce_max(self.ptr(), self.numel(), self.dtype.fmt), (), self.dtype.fmt)
    else: raise NotImplementedError("CUDA ops not implemented yet")
  def reduce_min(self):
    if self.__device.type_ == "cpu": return cpu.data(cpu_ops.reduce_min(self.ptr(), self.numel(), self.dtype.fmt), (), self.dtype.fmt)
    else: raise NotImplementedError("CUDA ops not implemented yet")
  def max(self, other): pass


__all__ = ["Buffer"]
