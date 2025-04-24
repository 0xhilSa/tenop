#from rush.csrc import cuda_core, cpu_core
from rush.engine import cpu_core
from rush import dtypes

#x = [1,2,3,4,5,6]
#device_ptr = cuda_core.toCUDA(x, dtypes.ulonglong.fmt)
#print(device_ptr)
#print(cuda_core.toList(device_ptr, len(x), dtypes.ulonglong.fmt))
#host_ptr = cuda_core.toCPU(device_ptr, len(x), dtypes.ulonglong.fmt)
#print(cpu_core.toList(host_ptr, len(x), dtypes.ulonglong.fmt))

x = 2
ptr = cpu_core.s_toCPU(x, dtypes.float32.fmt)
print(ptr)
print(cpu_core.s_fromCPU(ptr, 'f'))
