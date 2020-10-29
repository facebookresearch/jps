import os
import sys
import pickle
import json
import argparse
import pandas as pd

def process_policy(policy, infoset, reach, distribution):
    player, card, history = infoset.split("-")
    key = (int(player[1]), int(card.split("=")[1]), history)

    probmax = 0
    opt_a = None
    for a, prob in distribution.items():
        if prob > probmax:
            probmax = prob
            opt_a = a
    
    policy[key] = (reach, opt_a)


def load_policies(filename):
    prompt = "json_str: "

    state = 0
    curr_policy = dict()
    last_infoset = None
    last_distribution = dict()

    policies = []

    for line in open(filename, "r"):
        if line.startswith("Player 1 strategy"):
            state = 1
        elif line.startswith("Player 2 strategy"):
            state = 2
        elif line.startswith(prompt):
            items = json.loads(line[len(prompt):])
            scores = { k: float(v) for k, v in items.items() }
        elif line.startswith("Time spent: "):
            process_policy(curr_policy, last_infoset, last_reach, last_distribution)
            policies.append(dict(scores=scores, policy=curr_policy))
            state = 0
            last_infoset = None
            curr_policy = dict()
        elif state > 0:
            if line.startswith("P"):
                if last_infoset is not None:
                    process_policy(curr_policy, last_infoset, last_reach, last_distribution)
                last_infoset, last_reach = line.split(",")
                last_reach = float(last_reach.split(":")[1])
                last_distribution = dict()
            elif line != "\n":
                action, prob = line.split(":")
                action = eval(action)[0]
                prob = float(prob)
                last_distribution[action] = prob

    return policies

def print_policy_table(policy):
    # Get max c1 and c2
    maxcards = [0, 0]
    for (player, card, history), p in policy.items():
        maxcards[player - 1] = max(maxcards[player - 1], card) 

    table = []
        
    for c1 in range(maxcards[0] + 1):
        this_table = dict()
        for c2 in range(maxcards[1] + 1):
            history = "r"
            player = 1
            while True:
                c = c1 if player == 1 else c2
                key = (player, c, history)

                p = policy.get(key, None)

                if p is None:
                    break

                opt_a = p[1]
                history += str(opt_a)
                player = 3 - player

            if history[-1] == 'r':
                reward = 0
            else:
                contract = 1 << (int(history[-2]) - 1)
            if c1 + c2 >= contract:
                reward = contract
            else:
                reward = 0

            this_table[c2] = history[1:] + " (" + str(reward) + ")"
        table.append(this_table)

    print(pd.DataFrame(table))

def compare_dict(v1, v2):
    s = []

    all_keys = set(list(v1.keys()) + list(v2.keys()))

    for k in all_keys:
        vv1 = v1.get(k, 0)
        vv2 = v2.get(k, 0)
        if vv2 != vv1:
            s.append(f"{k}: {vv1} != {vv2}")
    return s

def compare_policy(p1, p2):
    ret = ""
    all_keys = set(list(p1.keys()) + list(p2.keys()))
    for k in all_keys:
        reach1, v1 = p1.get(k, (0.0, {}))
        reach2, v2 = p2.get(k, (0.0, {}))
        s = compare_dict(v1, v2)
        if len(s) > 0:
            ret += f"{k}: ({reach1}, {reach2}) \n" + "\n".join(s) + "\n"

    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--log', type=str)
    parser.add_argument('--dump_pkl', type=str, default=None)
    parser.add_argument('--k', type=int, default=None)

    args = parser.parse_args()

    policies = load_policies(args.log)
    if args.dump_pkl:
        pickle.dump(policies, open(args.dump_pkl, "wb"))

    # policies = pickle.load(open(sys.argv[1], "rb"))
    opt_k = None
    opt_policy = None
    opt_score = 0

    for k, p in enumerate(policies):
        this_score = p["scores"]["Search"] 
        if this_score > opt_score:
            opt_score = this_score
            opt_policy = p["policy"]
            opt_k = k

    print("Optimal policies")
    for k, p in enumerate(policies):
        this_score = p["scores"]["Search"] 
        if this_score == opt_score:
            print(f"score: {this_score}")
            print_policy_table(p["policy"])

    if args.k is not None:
        print("Chosen policy")

        policy = policies[args.k]["policy"]
        print(f"Policy: score: {policies[k]['scores']['Search']}")
        print_policy_table(policy)

        # print(f"{compare_policy(opt_policy, policy)}")


