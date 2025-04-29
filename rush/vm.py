from .buffers import Buffer
from typing import Union

class VirtualMemory:
  def __init__(self, size:Union[int,None]=None):
    self.__memory: dict[str, Buffer] = {}
  def register(self, label:str, buffer: Buffer):
    if label in self.__memory: raise ValueError(f"Buffer: '{label}' already exist")
    self.__memory[label] = buffer
  def delete(self, label):
    if label in self.__memory: del self.__memory[label]
  def get(self, label):
    if label not in self.__memory: raise ValueError(f"Label: '{label}' not found")
    return self.__memory[label]
