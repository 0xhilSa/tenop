from cuten import dtypes
from cuten.tensor import Tensor

x = [
      [[1,2,3,4],[5,6,7,8]],
      [[1,4,9,16],[25,36,49,64]]
    ]

a = Tensor(x, dtype=dtypes.complex128)
print(a)
print(a.shape)
print(a.numpy())

b = Tensor([1,2,3,4,5,6], dtypes.uint64, lazy=True)
print(b)
print(b.shape)
print(b.numpy())
