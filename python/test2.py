import torch

d = 10

pi = torch.rand(d)
pi /= pi.sum()

a = torch.randn(d).sort()[0] * 2

bs = torch.arange(-5, 5, step=0.01)
bs = torch.cat([bs, a])

ratios = torch.zeros_like(bs)

for i, b in enumerate(bs):
    delta = (a - b).abs() * pi
    ratios[i] = delta.sqrt().sum() / delta.sum().sqrt()

min_index = ratios.argmin()

print(f"{ratios[min_index]}, b = {bs[min_index]}")
print("a")
print(a)
    
print("pi")
print(pi)

print("ratios on grid")
print(ratios[-len(a):])

# print("bs")
# print(bs)
# print("ratios")
# print(ratios)
