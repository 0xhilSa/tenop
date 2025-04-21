from pynum.csrc import cuda_core, cpu_core
from pynum import dtypes

x = [1,2,3,4,5,6]
device_ptr = cuda_core.toCUDA(x, dtypes.complex256.fmt)
print(device_ptr)
print(cuda_core.toList(device_ptr, len(x), dtypes.complex256.fmt))
host_ptr = cuda_core.toCPU(device_ptr, len(x), dtypes.complex256.fmt)
print(cpu_core.toList(host_ptr, len(x), dtypes.complex256.fmt))
