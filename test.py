from tenop import Tensor
import tenop


a = Tensor([[1,2,0],[4,5,6]], dtype=tenop.boolean, device="cpu:0")
b = Tensor([[1,2,12],[4,5,6]], dtype=tenop.char, device="cpu:0")

x = Tensor([[1,2,12],[4,5,6]], dtype=tenop.longlong, device="cpu:0")
y = Tensor([[1,2,12],[4,5,6]], dtype=tenop.ulonglong, device="cpu:0")

q = Tensor([[1,2,12],[4,5,6]], dtype=tenop.float32, device="cpu:0")
w = Tensor([[1,2,12],[4,5,6]], dtype=tenop.float64, device="cpu:0")
e = Tensor([[1,2,12],[4,5,6]], dtype=tenop.longdouble, device="cpu:0")

c = Tensor([[1+4j,1-3j,-3+7j],[-1-0.7j,7j,1]], dtype=tenop.complex128, device="cpu:0") # min, max func won't work

print("================Tensor:a================")
print(a)
print(a.numpy())
print(a.reduce_max().numpy())
print(a.reduce_min().numpy())

print("================Tensor:b================")
print(b)
print(b.numpy())
print(b.reduce_max().numpy())
print(b.reduce_min().numpy())

print("================Tensor:x================")
print(x)
print(x.numpy())
print(x.reduce_max().numpy())
print(x.reduce_min().numpy())

print("================Tensor:y================")
print(y)
print(y.numpy())
print(y.reduce_max().numpy())
print(y.reduce_min().numpy())

print("================Tensor:q================")
print(q)
print(q.numpy())
print(q.reduce_max().numpy())
print(q.reduce_min().numpy())

print("================Tensor:w================")
print(w)
print(w.numpy())
print(w.reduce_max().numpy())
print(w.reduce_min().numpy())

print("================Tensor:e================")
print(e)
print(e.numpy())
print(e.reduce_max().numpy())
print(e.reduce_min().numpy())
