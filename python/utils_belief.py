import sys

from belief import *
sys.path.append("./belief")

sys.path.append("../dds")
import card_utils
from sample_utils import DealWalk, get_future_tricks, print_simulation

import torch
import re
import json
import math

belief_model_types = dict(
    vae=VAEModel,
    baseline=BaselineModel,
    lstm=LSTMModel,
    dyn=DynamicalOrderModel,
    direct=OpeningLeadModel,

    vae_old=VAEModelOld,
    baseline_old=BaselineModelOld,
    lstm_old=LSTMModelOld,
    dyn_old=DynamicalOrderModelOld,
    direct_old=OpeningLeadModelOld
)

def create_belief_model(model_type, model_name):
    model_class = belief_model_types.get(model_type, None)

    if model_class is None:
        belief_model = None
    else:
        model_params = torch.load(model_name)
        belief_model = model_class(load_model=model_name)
        belief_model.eval()
        belief_model.share_memory()

    return belief_model

def create_direct_model(model_name):
    direct_model = None
    if model_name is not None:
        direct_model_class = MeanTrickModel
    else:
        direct_model_class = None

    if direct_model_class is not None:
        direct_model_params = torch.load(model_name)
        direct_model = direct_model_class(load_model=model_name)
        direct_model.eval()
        direct_model.share_memory()

    return direct_model

def create_weight_model(model_name):
    weight_model = None
    if model_name is not None:
        weight_model_class = LogProbModel
    else:
        weight_model_class = None

    if weight_model_class is not None:
        weight_model_params = torch.load(model_name)
        weight_model = weight_model_class(load_model=model_name)
        weight_model.eval()
        weight_model.share_memory()

    return weight_model


def add_stats(stats, this_stats):
    if stats is None:
        stats = this_stats
    else:
        for k, v in this_stats.items():
            stats[k] += v
    return stats

def add_if_not_exist(d, key, v):
    if key not in d:
        d[key] = v
    else:
        d[key] += v

def accu_stats(d):
    '''
        d["stats.-posterior.1"] = { "a" : 2, "b": 3 }
        d["stats.-posterior.2"] = { "a" : 4, "b": 5 }
        d["stats.n"] = 2
        => 
        stats["a-posterior"] = 2 + 4
        stats["b-posterior"] = 3 + 5
        stats["n"] = 2
    '''
    stats = dict()

    for k, v in d.items():
        if k.startswith("stats."):
            items = k.split(".")
            suffix = items[1]

            if isinstance(v, dict):
                # suffix.
                for kk, vv in v.items():
                    key = f"{kk}{suffix}"
                    add_if_not_exist(stats, key, vv)
            else:
                add_if_not_exist(stats, suffix, v)

    return stats

def get_stats(stats):
    n = stats["n"]

    s = ""
    s += f"n: {n}\n"

    for k in sorted(stats.keys()):
        if k == "n":
            continue

        stats[k] /= n
        s += f"{k}: {stats[k]}\n"

    return s

    '''
    for rank, sel in rank_checks.items():
        print(f"{rank}-accu: {np.sum(stats['accu'][sel]) / len(sel)}")

    for i, suit in enumerate(suit_checks.keys()):
        print(f"{suit} len diff: {stats['suit_len_diff'][i]}")

    print(f"Hand length diff: {np.mean(stats['hand_len_diff'])}")
    '''

def batch_unsqueeze(b):
    for k, v in b.items():
        v.unsqueeze_(0)
    return b

def batch_squeeze(b):
    for k, v in b.items():
        v.squeeze_(0)
    return b

def batch_dup(n, t):
    if t.size(0) != 1:
        t = t.unsqueeze(0)
    sz = list(t.size())
    sz[0] = n
    return t.expand(*sz)

def compare_tensor_dict(d1, d2):
    assert len(d1) == len(d2)
    for k, v in d1.items():
        assert k in d2
        diff = (d2[k] - d1[k]).norm()
        if diff > 1e-6:
            print(f"key {k} not consistent!")
            import pdb
            pdb.set_trace()

power_matcher = re.compile(r"rank-prob-pow-([\.\d]+)")
exp_matcher = re.compile(r"rank-prob-exp-([\.\d]+)")

class BeliefSampler:
    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose

    def locate(self, idx, more_info={}):
        self.env.reset_to(idx, more_info.get("dealer", -1), more_info.get("vul", -1))
        data = json.loads(self.env.curr_json_str())
        data.update(more_info)

        for tbl in range(2):
            curr_bidd = data["bidd"][tbl]["seq"]

            for bid in curr_bidd:
                action = card_utils.bidStr2idx(bid)
                self.env.step(action)

            assert self.env.subgame_end(), "Subgame should end here!"

        if self.verbose:
            if "state_display" in data:
                print(data["state_display"])

            print(f"Dealer: {data['dealer']}")
            print(f"PBN: {data['pbn']}")

        self.data = data
        self.idx = idx

    def set_tbl(self, tbl):
        curr_bidd = self.data["bidd"][tbl]["seq"]

        strain, declarer = card_utils.get_contract_declarer(self.data["dealer"], curr_bidd)
        if declarer is None:
            '''
            print("No declare was made!")
            print(f"PBN: {data['pbn']}")
            print(f"Bidding table {tbl}")
            print(f"Bidd seq: {' '.join(curr_bidd)}")
            print(f"=== Table {tbl} ===")
            '''
            return False

        if self.verbose:
            print(f"Bidding table {tbl}")
            print(f"Bidd seq: {' '.join(curr_bidd)}")
            print(f"=== Table {tbl} ===")
            print(f"Declarer: {declarer}, Contract: {strain}")

        self.strain = strain
        self.declarer = declarer
        self.strain_idx = card_utils.STRAIN2IDX[strain]
        self.tbl = tbl
        self.curr_bidd = curr_bidd
        return True

    def set_opening_lead(self):
        # we only check the seat who performs the opening lead, (declarer + 1)
        self.seat = (self.declarer + 1) % card_utils.NUM_PLAYERS

        # f = env.feature_with_table_seat(tbl, seat, True)
        # (tbl, True, False): 
        #       True: contain ground truth card info
        #       False:ddo not contain debug info. 
        f = self.env.feature_opening_lead(self.tbl, True, False)
        batch_unsqueeze(f)

        self.f = f
        # The feature is always extracted from the current player's point of view (which is the player about to play).
        self.first = 0

    def set_seat(self, seat):
        self.seat = seat
        f = self.env.feature_with_table_seat(self.tbl, seat, True)
        batch_unsqueeze(f)
        self.f = f

        self.first = None
        raise NotImplementedError("set_seat is not done yet! Need to check the property of `feature_with_table_seat` in C++ func")

    def get_deals_from_prior(self, direct_model):
        with torch.no_grad():
            tricks = direct_model.mean_tricks(self.f)
        tricks = tricks.squeeze(0).cpu().numpy()
        return tricks

    def sample_deals(self, belief_model, n):
        f = self.f
        with torch.no_grad():
            init_rep = belief_model.get_init_rep(f)

        init_rep_b = batch_dup(n, init_rep)
        true_cards_b = batch_dup(n, f["cards"])

        deals = []
        n_sample = 0

        cards_data = []
        probs_data = []

        while len(deals) < n:
            with torch.no_grad():
                res = belief_model.sample(init_rep_b, true_cards_b)

            cards = res["cards"]
            probs = res["probs"]

            for i in range(n):
                card_map = ((cards[i] + self.first) % card_utils.NUM_PLAYERS).tolist()
                n_sample += 1
                try:
                    new_deal = DealWalk.from_card_map(card_map)
                    # Note that the model might generate invalid hands, so we need to do reject sampling.
                    deals.append(new_deal)
                    cards_data.append(cards[i])
                    probs_data.append(probs[i])
                    if len(deals) == n:
                        break
                except:
                    pass

        cards = torch.stack(cards_data, dim=0)
        probs = torch.stack(probs_data, dim=0)

        success_rate = n / n_sample
        return dict(idx=self.idx, declarer=self.declarer, strain=self.strain, strain_idx=self.strain_idx, tbl=self.tbl, deals=deals, cards=cards, probs=probs, success_rate=success_rate)

    def get_weights(self, args, weight_model, samples):
        n = len(samples["deals"])

        weights = torch.ones(n, device=samples["cards"][0].device)
        heu = args.weight_heuristics

        probs = samples["probs"]
        logprobs = torch.sum(probs.log(), dim=1)

        if weight_model is None:
            if heu is None or not heu.startswith("rank-prob"):
                return weights / n

            _, indices = torch.sort(logprobs, descending=True)
            m = power_matcher.match(heu)
            if m:
                p = float(m.group(1))
                for rank, i in enumerate(indices):
                    weights[i] = 1.0 / pow(rank + 1, p)
            else:
                m = exp_matcher.match(heu)
                if m:
                    p = float(m.group(1))
                    for rank, i in enumerate(indices):
                        weights[i] = math.exp(-rank * p)
                else:
                    raise RuntimeError(f"Unknown heuristics: {heu}")
        else:
            # When there is weight model.
            assert self.first == 0
            # s = features["s"].expand(n, -1).contiguous().view(1, n, -1)
            cur_features = dict(cards=samples["cards"], s=self.f["s"].expand(n, -1))
            with torch.no_grad():
                log_rho = weight_model(cur_features).squeeze(1)
            weights = (log_rho - logprobs)

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = weights.detach().cpu().numpy()

        # s1 = np.sum(weights)
        # s2 = np.sum(weights * weights)
        # print(f"effective sample rate = {s1 * s1 / s2}")

        return weights

    def get_gt_sample(self):
        assert self.first == 0
        deal_gt = [ DealWalk.from_card_map(self.f["cards"].squeeze().tolist()) ]
        future_tricks_gt, _ = get_future_tricks(deal_gt, self.strain_idx, self.first)
        future_tricks_gt = future_tricks_gt[0]
        return dict(deal=deal_gt, future_tricks_gt=future_tricks_gt)


