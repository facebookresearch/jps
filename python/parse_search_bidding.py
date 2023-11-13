import re
import sys
import json
import pickle
import argparse
from collections import defaultdict, OrderedDict
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument("--input", type=str)
parser.add_argument("--filename", type=str, default="bridge_record.pkl")
parser.add_argument("--reload", action="store_true")

args = parser.parse_args()

if args.reload:
    s = "json_state: "
    records = []
    for line in open(args.input, "r"):
        if not line.startswith(s):
            continue
        records.append(json.loads(line[len(s):]))

    pickle.dump(records, open(args.filename, "wb"))
else:
    records = pickle.load(open(args.filename, "rb"))

def remove_bracket(s):
    if s.startswith('(') and s.endswith(")"):
        return s[1:-1]
    else:
        return s

# Check bidding sequence and plot their reward distribution. 
bidds = defaultdict(list)
first_n = 4


length = np.zeros(40)
bids = {}
final_bids = {}

def inc(d, a):
    if a in d:
        d[a] += 1
    else:
        d[a] = 1

for record in records:
    # key

    n0 = len(record["bidd"][0]["seq"])
    n1 = len(record["bidd"][1]["seq"])
    seq0 = [ remove_bracket(record["bidd"][0]["seq"][i]) for i in range(n0) ]
    seq1 = [ remove_bracket(record["bidd"][1]["seq"][i]) for i in range(n1) ]

    length[n0] += 1
    length[n1] += 1

    for k in seq0:
        inc(bids, k)
    for k in seq1:
        inc(bids, k)

    for j in range(n0 - 1, 0, -1):
        if seq0[j] not in ["P", "X", "XX"]:
            inc(final_bids, seq0[j])
    for j in range(n1 - 1, 0, -1):
        if seq1[j] not in ["P", "X", "XX"]:
            inc(final_bids, seq1[j])

print(length)
print(bids)
print(final_bids)

'''
    key = "-".join(seq0) + "_" + "-".join(seq1)

    bidds[key].append(record)

bidds = sorted(list(bidds.items()), key=lambda x: -len(x[1]))

print(f"#different_keys: {len(bidds)}")

for k, v in bidds:
    if len(v) > 5:
        r = [ vv["reward"] for vv in v ]
        print(f"{k}: cnt: {len(r)}, arg: {sum(r) / len(r):.3f}, min: {min(r):.3f}, max: {max(r):.3f}")
        for vv in v:
            print("=====Start=======")
            print(f"Dealer: {vv['dealer']}")
            print(vv["state_display"])
            print(vv["reward"])
            print("===End====")
'''
    
