from types import NoneType
from typing import Type, Union, Sequence
from .buffers import Buffer, LazyBuffer
from .dtypes import DType

Scalar = Union[int, float, complex, bool, DType]
TensorType = Union[Scalar, Sequence["TensorType"]]

class Tensor:
  def __init__(self, tensor:TensorType, dtype:Union[Type,DType,NoneType]=None, device:str="cpu:0", requires_grad:bool=False, const:bool=False, lazy:bool=False):
    if lazy: self.__buffer = LazyBuffer(tensor, dtype, device, requires_grad, const)
    else: self.__buffer = Buffer(tensor, dtype, device, requires_grad, const)
  def __repr__(self): return f"<Tensor({self.__buffer})>"
  @property
  def device(self): return self.__buffer.device
  @property
  def dtype(self): return self.__buffer.dtype
  @property
  def ndim(self): return self.__buffer.ndim
  @property
  def strides(self): return self.__buffer.stride
  def requires_grad(self): return self.__buffer.requires_grad()
  def is_const(self): return self.__buffer.is_const()
  def numel(self): return self.__buffer.numel()
  def shape(self, dim:Union[int,None]=None):
    if dim is None: return self.__buffer.shape
    if not 0 <= dim < len(self.__buffer.shape): raise IndentationError("Index out of range")
    return self.__buffer.shape[dim]
  def sizeof(self): return self.__buffer.sizeof()
  def numpy(self): return self.__buffer.numpy()
