import torch
import random

def compute_best_act(r, S):
    v = torch.zeros(r.size(1))
    for s in S:
        v += r[s, :] 
        
    return v.max()

# reward matrix.
num_a1 = 8
num_a2 = 8
num_s1 = 8 
r = torch.zeros(num_s1, num_a2)
# Identity case.
for i in range(num_s1):
    r[i, i] = 1.0

map_a1_s1 = [ set() for _ in range(num_a1) ]
map_s1_a1 = [None] * num_s1

# Random assignment.  
for i in range(num_s1):
    a = random.randint(0, num_a1 - 1)
    map_a1_s1[a].add(i)
    map_s1_a1[i] = a

# compute reward for each bin
values_a1 = [None] * num_a1
for j in range(num_a1):
    print(map_a1_s1[j])
    import pdb
    pdb.set_trace()
    max_value, _ = compute_best_act(r, map_a1_s1[j])
    values_a1[j] = max_value.item()

# Start the loop
while True:
    # Pick one state, move it from one to the other.
    s1 = random.randint(0, num_s1 - 1)
    a1 = map_s1_a1[s1]

    # value after removing the state from that bin.
    subset = map_a1_s1[a1] - s1
    value_after_removal, max_indices = compute_best_act(r, subset)

    global_delta = value_after_removal - values_a1[a1]

    values_after_addition = torch.zeros(num_a1)

    for j in range(num_a1):
        if j == map_s1_a1[s1]: continue

        new_j = map_a1_s1[j] + s1
        values_after_addition[j], _ = compute_best_act(r, new_j)
    
    deltas = global_delta + values_after_addition - values_a1
    deltas[a1] = 0

    # Find best target.
    max_value, max_index = deltas.max()
    print(f"Delta: {max_value.item()}")

    max_index = max_index.item()
     
    # Then move the state from old to new. 
    if max_index != a1:
        map_a1_s1[a1] -= s1
        map_a1_s1[max_index] += s1

        map_s1_a1[s1] = max_index

        values_a1[a1] = value_after_removal
        values_a1[max_index] = values_after_addition[max_index]
     

        
            



