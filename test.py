from rush import dtypes
from rush.buffers import Buffer


buff1 = Buffer([1,2,3,4,5], device="cpu", dtype=dtypes.complex128)
buff2 = Buffer([1,2,3,4,5], device="cuda", dtype=dtypes.complex128)
print(buff1)
print(buff1.pointer)
print(buff1.length)
print(buff1.dtype)
print(buff1.device)
print(buff1.backend)

print(buff2)
print(buff2.pointer)
print(buff2.length)
print(buff2.dtype)
print(buff2.device)
print(buff2.backend)


