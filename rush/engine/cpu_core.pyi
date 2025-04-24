from typing import Any, Type, Union
from ..dtypes import DType, Fmts
import ctypes


def s_toCPU(scalar:Any,fmt:Fmts) -> ctypes.c_void_p: ...
def s_fromCPU(pointer:ctypes.c_void_p, fmt:Fmts) -> Type: ...
