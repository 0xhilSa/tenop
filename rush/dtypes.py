from dataclasses import dataclass
from typing import Literal

Fmts = Literal["?", "b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d", "g", "F", "D", "G"]

@dataclass(frozen=True, eq=True)
class DType:
  ctype:str
  signed:bool
  fmt:Fmts
  size:int
  @classmethod
  def new(cls, ctype:str, signed:bool, fmt:Fmts, size:int): return DType(ctype, signed, fmt, size)
  def __repr__(self): return f"<DType(ctype='{self.ctype}', signed={self.signed}, fmt='{self.fmt}', size={self.size})>"
  @property
  def nbit(self): return self.size * 8
  @property
  def nbyte(self): return self.size

bool_ = boolean = DType.new("bool", False, "?", 1)

byte = char = int8 = DType.new("char", True, "b", 1)
short = int16 = DType.new("short", True, "h", 2)
int_ = int32 = DType.new("int", True, "i", 4)
long = int64 = DType.new("long", True, "l", 8)
longlong = DType.new("long long", True, "q", 8)

ubyte = uchar = uint8 = DType.new("unsigned char", False, "B", 1)
ushort = uint16 = DType.new("unsigned short", False, "H", 2)
uint = uint32 = DType.new("unsigned int", False, "I", 4)
ulong = uint64 = DType.new("unsigned long", False, "L", 8)
ulonglong = DType.new("unsigned long long", False, "Q", 8)

float_ = float32 = DType.new("float", True, "f", 4)
double = float64 = DType.new("double", True, "d", 8)
longdouble = float128 = DType.new("long double", True, "g", 16)

floatcomplex = complex64 = DType.new("float _Complex", True, "F", 8)
doublecomplex = complex128 = DType.new("double _Complex", True, "D", 16)
longdoublecomplex = complex256 = DType.new("long double _Complex", True, "G", 32)
