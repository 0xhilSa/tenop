from cuten import dtypes
from cuten.tensor import Tensor
import timeit

start = timeit.default_timer()
print("----------Tensor:a----------")
a = Tensor([[[1,2,3,4],[5,6,7,8]],[[1,4,9,16],[25,36,49,64]]], dtype=dtypes.complex128, device="cuda:0")
print(a)
print(a.sizeof())
print(a.shape())
print(a.numel())
print(a.numpy())

print("----------Tensor:b----------")
b = Tensor([1,2,3,4,5,6], dtypes.uint64, device="cuda:0", lazy=True)
print(b)
print(b.sizeof())
print(b.shape())
print(b.numel())
print(b.numpy())

print("----------Tensor:c----------")
c = Tensor(2, dtypes.uint64, device="cuda:0", lazy=True)
print(c)
print(c.sizeof())
print(c.shape())
print(c.numel())
print(c.numpy())
end = timeit.default_timer()

print("----------Time-----------")
print(end - start)
