from typing import List, Any
from ..dtypes import Fmts
import ctypes


def toCPU(array1D:List[Any], fmt:Fmts) -> ctypes.c_void_p: ...
def toList(pointer:ctypes.c_void_p, length:int, fmt:Fmts) -> List[Any]: ...
