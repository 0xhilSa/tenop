from typing import Tuple

class Shape:
  def __init__(self, shape:Tuple[int, ...]):
    self.__shape = list(shape)
  def __repr__(self): return f"Shape({self.__shape})"
  def __len__(self): return len(self.__shape)
  def tolist(self): return self.__shape
  def totuple(self): return tuple(self.__shape)
  def __iter__(self): return iter(self.__shape)
  def __getitem__(self,index:int):
    if not 0 <= index < len(self): raise IndexError("Index out of range")
    return self.__shape[index]
  def numel(self):
    elements = 1
    for dim in self: elements *= dim
    return elements
