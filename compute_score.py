from bisect import bisect
import statistics
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_file", type=str, default="./logs/jps_14days.log")
parser.add_argument("--dds_file", type=str, default="./logs/against_WBridge5.raw")
args = parser.parse_args()

f = open(args.log_file)
dds = open(args.dds_file)

f_lines = f.readlines()
dds_lines = dds.readlines()
f.close()
dds.close()

vuls = ["None", "NS", "EW", "Both"]
strains = ["C", "D", "H", "S", "N"]

strain_map = {}
for i in range(5):
    strain_map[strains[i]] = i 
vul_map = {}
for i in range(4):
    vul_map[vuls[i]] = i  

def score(tricks, level, strain, doubled, vul):
    """Score for a contract for a given number of tricks taken."""
    target = level + 6
    overtricks = tricks - target
    if overtricks >= 0:
        per_trick = 20 if strain in ["C", "D"] else 30
        base_score = per_trick * level
        bonus = 0
        if strain == "N":
            base_score += 10
        if doubled == 1:
            base_score *= 2
            bonus += 50
        if doubled == 2:
            base_score *= 4
            bonus += 100
        bonus += [300, 500][vul] if base_score >= 100 else 50
        if level == 6:
            bonus += [500, 750][vul]
        elif level == 7:
            bonus += [1000, 1500][vul]
        if not doubled:
            per_overtrick = per_trick
        else:
            per_overtrick = [100, 200][vul] * doubled
        overtricks_score = overtricks * per_overtrick
        return base_score + overtricks_score + bonus
    else:
        if not doubled:
            per_undertrick = [50, 100][vul]
            return overtricks * per_undertrick
        else:
            if overtricks == -1:
                score = [-100, -200][vul]
            elif overtricks == -2:
                score = [-300, -500][vul]
            else:
                score = 300 * overtricks + [400, 100][vul]
        if doubled == 2:
            score *= 2
        return score

def process(line, dds_line, dealer, vul, ttt, raw, table):
    tokens = line.split()[4:-3]
    for token in tokens:
        if token[0] == "(":
            token = token[1:-1]
    dds_tokens = dds_line.split()
    doubled = 0
    contract = ""
    old_declarer = -1
    new_declarer = -1
    player = (dealer + len(tokens)) % 4
    for token in reversed(tokens):
        if token[0] == "(":
            token = token[1:-1]
        player = (player + 3) % 4
        if token == "XX":
            doubled = 2
            continue
        elif token == "X":
            if doubled == 0:
                doubled = 1
            continue
        elif token == "P":
            continue
        else:
            contract = token
            old_declarer = player
            break
    
    #All pass
    if contract == "":
        assert(raw == 0)
        return 0

    player = (dealer + 3) % 4

    for token in tokens:
        if token[0] == "(":
            token = token[1:-1]
        player = (player + 1) % 4
        if (((player - old_declarer) % 2 == 0) and (token[-1] == contract[-1])):
            new_declarer = player
            break
    
    strain_idx = strain_map[contract[-1]]
    old_ttt = int(dds_tokens[strain_idx * 4 + old_declarer])
    new_ttt = int(dds_tokens[strain_idx * 4 + new_declarer])

    assert(old_ttt == ttt)
    vul_idx = vul_map[vul]

    v = 0
    if old_declarer % 2 == 0 and vul_idx == 1:
        v = 1
    if old_declarer % 2 == 1 and vul_idx == 2:
        v = 1   
    if vul_idx == 3:
        v = 1

    old_reverse_score = 1
    new_reverse_score = 1
    if old_declarer % 2 == 1:
        old_reverse_score *= -1
    if new_declarer % 2 == 1:
        new_reverse_score *= -1

    old_raw = old_reverse_score * score(old_ttt, int(contract[0]), contract[-1], doubled, v)
    new_raw = new_reverse_score * score(new_ttt, int(contract[0]), contract[-1], doubled, v)

    assert(old_raw == raw)

    return new_raw

def imps(my, other):
    imp_table = [
        15, 45, 85, 125, 165, 215, 265, 315, 365, 425, 495, 595, 745, 895,
        1095, 1295, 1495, 1745, 1995, 2245, 2495, 2995, 3495, 3995]
    return bisect(imp_table, abs(my - other)) * (1 if my > other else -1)

new_r = []
N = 1000
for i in range(N):
    idx = i * 18
    dealer = int(f_lines[idx].split()[2])
    vul = f_lines[idx].split()[4][:-1]
    ttt0 = int(f_lines[idx + 14].split()[6][:-1])
    ttt1 = int(f_lines[idx + 15].split()[6][:-1])
    raw0 = int(f_lines[idx + 14].split()[8])
    raw1 = int(f_lines[idx + 15].split()[8])

    assert f_lines[idx + 1] == dds_lines[i * 2], f"dds record and log record doesn't match! dds deal: {dds_lines[i*2]}, f deal: {f_lines[idx+1]}"

    res0 = process(f_lines[idx + 10], dds_lines[i * 2 + 1], dealer, vul, ttt0, raw0, 0)
    res1 = process(f_lines[idx + 11], dds_lines[i * 2 + 1], dealer, vul, ttt1, raw1, 1)
    new_reward = imps(res0, res1)
    new_r.append(new_reward)
    
print(f"mean = {sum(new_r) / N}, std = {statistics.stdev(new_r) / math.sqrt(N)}")
