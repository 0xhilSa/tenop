from __future__ import annotations
from typing import Literal, Type, Union
from dataclasses import dataclass

Fmts = Literal["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d", "g", "F", "D", "G"]
priority = {bool:1, int:2, float:3, complex:4}

@dataclass(frozen=True, eq=True)
class DType:
  fmt:Fmts
  ctype:str
  nbyte:int
  signed:bool
  @classmethod
  def new(cls, fmt:Fmts, ctype:str, nbyte:int, signed:bool): return DType(fmt, ctype, nbyte, signed)
  def __repr__(self): return f"<DType(ctype='{self.ctype}', fmt='{self.fmt}', nbyte={self.nbyte}, signed={self.signed})>"
  @property
  def nbit(self): return self.nbyte * 8
  @staticmethod
  def to_pytype(dtype:DType):
    if dtype.fmt in BOOLEAN: return bool
    elif dtype.fmt in (INT + UINT): return int
    elif dtype.fmt in FLOAT: return float
    elif dtype.fmt in COMPLEX: return complex
    else: raise TypeError
  @staticmethod
  def from_pytype(dtype:Type[Union[int, float, complex, bool]]):
    if dtype == bool: return DType.new("?", "bool", 1, False)
    elif dtype == int: return DType.new("l", "long", 8, True)
    elif dtype == float: return DType.new("d", "double", 8, True)
    elif dtype == complex: return DType.new("D", "double _Complex", 16, True)
    else: TypeError(f"Invalid Type: {dtype.__name__}")


bool_ = boolean = DType.new("?", "bool", 1, False)
char = int8 = DType.new("b", "char", 1, True)
uchar = uint8 = DType.new("B", "unsigned char", 1, False)
short = int16 = DType.new("h", "short", 2, True)
ushort = uint16 = DType.new("H", "unsigned short", 2, False)
int_ = int32 = DType.new("i", "int", 4, True)
uint = uint32 = DType.new("I", "unsigned int", 4, False)
long = int64 = DType.new("l", "long", 8, True)
ulong = uint64 = DType.new("L", "unsigned long", 8, False)
longlong = DType.new("q", "long long", 8, True)
ulonglong = DType.new("Q", "unsigned long long", 8, False)
float_ = float32 = DType.new("f", "float", 4, True)
double = float64 = DType.new("d", "double", 8, True)
longdouble = DType.new("g", "long double", 16, True)
floatcomplex = complex64 = DType.new("F", "float _Complex", 8, True)
doublecomplex = complex128 = DType.new("D", "double _Complex", 16, True)
longdoublecomplex = complex256 = DType.new("G", "long double _Complex", 32, True)

BOOLEAN = [boolean]
INT = [char, int8, short, int16, int_, int32, long, int64, longlong]
UINT = [uchar, uint8, ushort, uint16, uint, uint32, ulong, uint64, ulonglong]
FLOAT = [float32, float64, longdouble]
COMPLEX = [floatcomplex, complex64, doublecomplex, complex128, longdoublecomplex, complex256]

maps = {int:int64, float:float64, complex:complex128, bool:boolean}
