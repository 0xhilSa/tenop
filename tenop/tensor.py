from __future__ import annotations
from typing import Type, Tuple
from tenop.buffers import Buffer
from tenop.dtypes import DType
from .helpers import Scalar, TensorType

class Tensor:
  def __init__(self, buf:Scalar|TensorType, dtype:DType|Type|None=None, device:str="cpu:0", requires_grad:bool=False, const:bool=False):
    self.__buf = Buffer(buf, dtype=dtype, device=device, requires_grad=requires_grad, const=const)
    self.__dtype = self.__buf.dtype
    self.__device = self.__buf.device
    self.__requires_grad = self.__buf.requires_grad
    self.__const = self.__buf.isconst
    self.__shape = self.__buf.shape
    self.__ndim = self.__buf.ndim
  def __repr__(self): return f"<Tensor({self.__buf})>"
  @property
  def device(self): return self.__device
  @property
  def dtype(self): return self.__dtype
  @property
  def shape(self): return self.__shape
  @property
  def ndim(self): return self.__ndim
  @property
  def isconst(self): return self.__const
  @property
  def requires_grad(self): return self.__requires_grad
  def numel(self): return self.__buf.numel()
  def data(self): return self.__buf.data()
  def numpy(self): return self.__buf.numpy()
  def astype(self, dtype:DType): return Tensor(self.__buf.data(), dtype=dtype, device=str(self.device), requires_grad=self.requires_grad, const=self.isconst)
  def clone(self): return Tensor(self.__buf.data(), device=str(self.device), requires_grad=self.requires_grad, const=self.isconst)
  def const(self):
    if self.isconst: return self
    return Tensor(self.__buf.data(), dtype=self.dtype, device=str(self.device), requires_grad=self.requires_grad, const=True)
  def cuda(self):
    if self.device.type_ == "cuda": return self
    elif self.device.type_ == "cpu": return Tensor(self.__buf.data(), dtype=self.dtype, device=f"cuda:0", requires_grad=self.requires_grad, const=False)
  def cpu(self):
    if self.device.type_ == "cpu": return self
    elif self.device.type_ == "cuda": return Tensor(self.__buf.data(), device=f"cpu:0", requires_grad=self.requires_grad, const=False)
  @staticmethod
  def full(value, shape:Tuple[int,...], device:str="cpu:0", requires_grad:bool=False, const:bool=False): return Tensor(Buffer.full(value, shape), device=device, requires_grad=requires_grad, const=const)
  @staticmethod
  def ones(shape:Tuple[int, ...], device:str="cpu:0", requires_grad:bool=False, const:bool=False): return Tensor(Buffer.full(1, shape), device=device, requires_grad=requires_grad, const=const)
  @staticmethod
  def zeros(shape:Tuple[int, ...], device:str="cpu:0", requires_grad:bool=False, const:bool=False): return Tensor(Buffer.full(0, shape), device=device, requires_grad=requires_grad, const=const)
  def reduce_max(self): return Tensor(self.__buf.reduce_max(), device=str(self.device), requires_grad=self.requires_grad, const=self.isconst)
  def reduce_min(self): return Tensor(self.__buf.reduce_min(), device=str(self.device), requires_grad=self.requires_grad, const=self.isconst)
