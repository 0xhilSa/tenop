from rush.engine import cpu_core, cuda_core
from rush import dtypes


x = 23
dptr = cuda_core.s_toCUDA(x, dtypes.uint8.fmt)
print(dptr)
print(cuda_core.s_fromCUDA(dptr, dtypes.uint8.fmt))
hptr = cuda_core.s_toHost(dptr, dtypes.uint8.fmt)
print(hptr)
print(cpu_core.s_fromCPU(hptr, dtypes.uint8.fmt))

x = [1,2,3,4,5,6,7]
dptr = cuda_core.v_toCUDA(x, dtypes.uint8.fmt)
print(dptr)
print(cuda_core.v_fromCUDA(dptr, len(x), dtypes.uint8.fmt))
hptr = cuda_core.v_toHost(dptr, len(x), dtypes.uint8.fmt)
print(hptr)
print(cpu_core.v_fromCPU(hptr, len(x), dtypes.uint8.fmt))

y = 12
hhptr = cpu_core.s_toCPU(y, dtypes.longdoublecomplex.fmt)
print(hhptr)
print(cpu_core.s_fromCPU(hhptr, dtypes.longdoublecomplex.fmt))
