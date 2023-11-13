import torch
import torch.nn.functional as F

@torch.jit.script
def multinomial_test(prob):
    idx = prob.multinomial(1)
    val = prob.gather(1, idx)
    if val.min() <= 0:
        print(f"idx = {idx}")
        print(f"val = {val}")

prob = torch.randint(52, (128,), device="cuda")
prob = F.one_hot(prob, 52).float()
num = 100000

for _ in range(num):
    multinomial_test(prob)
