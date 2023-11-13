import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, d):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(input_dim, hidden_dim)

        self.sample_head = nn.Linear(hidden_dim, d)
        self.prob_head = nn.Linear(hidden_dim, d)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        h = F.relu(self.linear(x))
        logpi = self.logsoftmax(self.sample_head(h))
        true_prob = self.prob_head(h)
        return dict(logpi=logpi, true_prob=true_prob)

    def loss(self, x, actions, probs):
        # x: input, 
        # a: action to be trained, 
        # probs: current probs for this action. 
        res = self.forward(x)

        err1 = F.nll_loss(res["logpi"], actions)
        sel_probs = res["true_prob"].gather(1, actions.unsqueeze(1))
        err2 = F.mse_loss(sel_probs, probs)

        return err1 + err2, dict(err1=err1.detach().item(), err2=err2.detach().item())

# A simple game.
# Sample in [0, d), if we get some bad examples (e.g,. 7, 9) then we get -10 reward, otherwise get 1 reward.
# Then we want to check the expectation of the reward. 
class Game:
    def __init__(self):
        self.d = 100

        self.rewards = torch.FloatTensor(self.d).fill_(1)
        self.rewards[7] = -30

        self.pi = torch.FloatTensor(self.d).fill_(1/self.d)
        self.mean_r = (self.pi * self.rewards).sum()

    def sample(self, N):
        samples = torch.multinomial(self.pi, N, replacement=True)
        probs = self.pi.gather(0, samples)
        return samples, probs

    def get_reward(self, samples):
        return self.rewards.gather(0, samples)

g = Game()

batchsize = 16

# Build a simple model to sample more on bad rewards.
input_dim = 2
hidden_dim = 2

model = Model(input_dim, hidden_dim, g.d)
x = torch.FloatTensor(batchsize, input_dim).fill_(0)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

counter = Counter()

for i in range(5000):
    samples, probs = g.sample(batchsize)
    rs = g.get_reward(samples)

    # Optimal resampling policy: p'(x) \propto p(x)|r(x)|
    # Since samples are already sampled from p(x), we only need to resample using |r(x)| 
    # don't need to normalize
    indices = torch.multinomial(rs.abs(), batchsize, replacement=True)

    sel_actions = samples[indices]
    sel_probs = probs[indices]

    for a in sel_actions:
        counter[a.item()] += 1

    optim.zero_grad()
    loss, stats = model.loss(x, sel_actions, sel_probs)

    loss.backward()
    optim.step()

    print(stats)

print(f"Final loss: {loss.detach().item()}")
print(counter.most_common(10))

K = 1000
avg_rs = torch.FloatTensor(K)
N = 10

print("Regular sampling")
for i in range(10):
    for k in range(K):
        samples, _ = g.sample(N)
        rs = g.get_reward(samples)
        avg_rs[k] = rs.mean() - g.mean_r
        
    print(f"avg_mean_diff: {avg_rs.mean()}, std: {avg_rs.std()}")

print("Trained sampling")
x = torch.FloatTensor(1, input_dim).fill_(0)

for i in range(10):
    for k in range(K):
        with torch.no_grad():
            res = model.forward(x)
        pi = res["logpi"].exp().squeeze()
        probs = res["true_prob"].squeeze()

        samples = torch.multinomial(pi, N, replacement=True) 
        sample_probs = pi.gather(0, samples)
        probs = probs.gather(0, samples)

        rs = g.get_reward(samples)
        avg_rs[k] = (rs * probs / sample_probs).mean() - g.mean_r

    print(f"avg_mean_diff: {avg_rs.mean()}, std: {avg_rs.std()}")
