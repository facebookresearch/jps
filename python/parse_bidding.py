import re
import sys
import json
import pickle
from collections import defaultdict, OrderedDict

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument("--input", type=str)
parser.add_argument("--use_n_record", type=int, default=None)
parser.add_argument("--reload", action="store_true")

parser.add_argument("--save_record", type=str, default="bridge_record.pkl")
parser.add_argument("--save_freq", type=str, default="bridge_freq.pkl")

parser.add_argument("--first_n_bid", type=int, default=4)
parser.add_argument("--n_bid", type=int, default=None)

args = parser.parse_args()

assert args.input.endswith(".bid"), "Input should ends with .bid"

prefix = args.input[:-4]
record_filename = prefix + ".record"
idx_filename = prefix + ".idx"

if args.reload:
    s = "json_state: "
    records = []
    for line in open(args.input, "r"):
        if not line.startswith(s):
            continue
        records.append(json.loads(line[len(s):]))
        records[-1]["idx"] = len(records) - 1
        if args.use_n_record is not None and args.use_n_record == len(records):
            break

    print(f"Save to {record_filename}")
    pickle.dump(records, open(record_filename, "wb"))
else:
    print(f"Load from {record_filename}")
    records = pickle.load(open(record_filename, "rb"))

def remove_bracket(s):
    if s.startswith('(') and s.endswith(")"):
        return s[1:-1]
    else:
        return s

# Check bidding sequence and plot their reward distribution. 
bidds = defaultdict(lambda : dict(stats=None, indices=list()))

for idx, record in enumerate(records):
    bidd0 = record["bidd"][0]["seq"]
    bidd1 = record["bidd"][1]["seq"]

    bidd0 = [ remove_bracket(b) for b in bidd0 ]
    bidd1 = [ remove_bracket(b) for b in bidd1 ]

    # key
    for n in range(1, args.first_n_bid + 1):
        seq0 = bidd0[:n]
        seq1 = bidd1[:n]

        # Joint key.
        key = "-".join(seq0) + "_" + "-".join(seq1)
        bidds[key]["indices"].append(idx)

        # Separate key
        key = "-".join(seq0)
        bidds[key]["indices"].append(idx)

        key = "-".join(seq1)
        bidds[key]["indices"].append(idx)

    if args.n_bid is not None:
        # Also add bid that are at intermediate locations.
        for i in range(1, len(bidd0)):
            for n in range(1, args.n_bid + 1):
                if i + n > len(bidd0):
                    continue
                seq = bidd0[i:i+n]
                key = "-" + "-".join(seq)
                bidds[key]["indices"].append(idx)

        for i in range(1, len(bidd1)):
            for n in range(1, args.n_bid + 1):
                if i + n > len(bidd1):
                    continue
                seq = bidd1[i:i+n]
                key = "-" + "-".join(seq)
                bidds[key]["indices"].append(idx)


for k, v in bidds.items():
    v["indices"] = set(v["indices"])
    r = [ records[idx]["reward"] for idx in v["indices"] ]
    v["stats"] = dict(length=len(r), avg=sum(r)/len(r), min=min(r), max=max(r), key=k)

print(f"#different_keys: {len(bidds)}")

bidds_list = sorted(list(bidds.items()), key=lambda x: -len(x[1]["indices"]))
for k, v in bidds_list:
    if len(v["indices"]) <= 5:
        break
    print(f"{k}: cnt: {len(r)}, arg: {sum(r) / len(r):.3f}, min: {min(r):.3f}, max: {max(r):.3f}")
    '''
    for idx in v["indices"]:
        record = records[idx]
        print("=====Start=======")
        print(f"Dealer: {record['dealer']}")
        print(record["state_display"])
        print(record["reward"])
        print("===End====")
    '''

print(f"Save to {idx_filename}")
pickle.dump(dict(bidds), open(idx_filename, "wb"))
    
