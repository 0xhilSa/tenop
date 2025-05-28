from __future__ import annotations
from typing import List, Tuple, Union

class Shape:
  def __init__(self, shape:Tuple[int, ...]):
    self.__shape = list(shape)
  def __repr__(self): return f"Shape({self.__shape})"
  def __len__(self): return len(self.__shape)
  def tolist(self): return self.__shape.copy()
  def totuple(self): return tuple(self.__shape)
  def __iter__(self): return iter(self.__shape)
  def __eq__(self, shape:Union[Shape,Tuple]):
    if not isinstance(shape, Shape): shape = Shape(shape)
    return self.__shape == shape.tolist()
  def __getitem__(self,index:int):
    if not 0 <= index < len(self): raise IndexError("Index out of range")
    return self.__shape[index]
  def numel(self):
    elements = 1
    for dim in self: elements *= dim
    return elements
