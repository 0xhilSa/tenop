from tenop import Tensor
import tenop


a = Tensor([[1,2,12],[4,5,6]], dtype=tenop.long, device="cpu:0")
b = Tensor([[1,2,12],[4,5,6]], dtype=tenop.ulong, device="cpu:0")

x = Tensor([[1,2,12],[4,5,6]], dtype=tenop.longlong, device="cpu:0")
y = Tensor([[1,2,12],[4,5,6]], dtype=tenop.ulonglong, device="cpu:0")

q = Tensor([[1,2,12],[4,5,6]], dtype=tenop.float32, device="cpu:0")
w = Tensor([[1,2,12],[4,5,6]], dtype=tenop.float64, device="cpu:0")
e = Tensor([[1,2,12],[4,5,6]], dtype=tenop.longdouble, device="cpu:0")

print(a)
print(a.reduce_max().numpy())

print(b)
print(b.reduce_max().numpy())

print(x)
print(x.reduce_max().numpy())

print(y)
print(y.reduce_max().numpy())

print(q)
print(q.reduce_max().numpy())

print(w)
print(w.reduce_max().numpy())

print(e)
print(e.reduce_max().numpy())
