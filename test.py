from rush import dtypes
from rush.buffers import Buffer
from rush.vm import VirtualMemory
vm = VirtualMemory()

buff1 = Buffer([1,2,3,4,5], device="cpu", dtype=dtypes.complex128)
buff2 = Buffer([1,2,3,4,5], device="cuda", dtype=dtypes.complex128)
buff3 = Buffer(2, device="cuda", dtype=dtypes.int64)
buff4 = Buffer(2, device="cpu", dtype=dtypes.int64)

vm.register("a", buff1)
vm.register("b", buff2)
vm.register("c", buff2)

print("===================vector-cpu====================")
print(buff1)
print(buff1.pointer)
print(buff1.length)
print(buff1.dtype)
print(buff1.device)
print(buff1.backend)
print(buff1.isscalar())
print(buff1.isvector())

print("===================vector-cuda====================")
print(buff2)
print(buff2.pointer)
print(buff2.length)
print(buff2.dtype)
print(buff2.device)
print(buff2.backend)
print(buff2.isscalar())
print(buff2.isvector())

print("===================scalar-cuda====================")
print(buff3)
print(buff3.pointer)
print(buff3.length)
print(buff3.dtype)
print(buff3.device)
print(buff3.backend)
print(buff3.isscalar())
print(buff3.isvector())

print("===================to-cuda====================")
tocuda1 = buff1.cuda()
print(tocuda1)
tocuda2 = buff4.cuda()
print(tocuda2)
