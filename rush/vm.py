from .buffers import Buffer
from typing import Union

class VirtualMemory:
  def __init__(self, size:Union[int,None]=None) -> None:
    self.__size = size
    self.__memory: dict[str, Buffer] = {}
  def register(self, label: str, buffer: Buffer) -> None:
    if label in self.__memory: raise ValueError(f"Buffer with label '{label}' already exists.")
    if self.__size is not None and len(self.__memory) >= self.__size: raise MemoryError(f"Memory limit of {self.__size} buffers exceeded.")
    self.__memory[label] = buffer
  def delete(self, label) -> None:
    if label in self.__memory: del self.__memory[label]
  def get(self, label) -> Buffer:
    if label not in self.__memory: raise ValueError(f"Label: '{label}' not found")
    return self.__memory[label]
