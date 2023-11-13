import sys
sys.path.append("..")

import pickle

import json
import os
import time
import torch
from copy import deepcopy
from datetime import datetime

import hydra

import tqdm

import numpy as np

import argparse
import re

from collections import defaultdict, OrderedDict

import math

import set_path
import random
set_path.append_sys_path()

import torch.multiprocessing as mp

from pybridge import factory

from a2c import A2CAgent

sys.path.append("../dds")
import card_utils
from sample_utils import DealWalk, get_future_tricks, print_simulation

sys.path.append("./belief")
import models

from utils_belief import *

from utils import *

def post_process(model_args):
    m = re.search(r"2020-(\d+)-(\d+)", model_args["load_model"])
    if m.group(1) == "01":
        print("Old model, adding extra parameters for loading.")
        model_args.update(dict(
            num_blocks=2, input_dim=268, num_action=39, hid_dim=200,
            max_importance_ratio=2,
            opp_hand_ratio=0,
            use_goal=False,
            use_old_feature=False,
            entropy_ratio=0.01,
            p_hand_loss_ratio=0.0,
            p_hand_loss_only_on_terminal=False,
            explore_ratio=0))

    return model_args


def print_tricks(tricks, title):
    print("*****************************************************************")
    print(f"{title}:")
    idx = np.where(tricks == np.max(tricks))
    tricks = np.reshape(tricks, (4, 13))
    print(tricks)
    print(f"{title}_max_idx = {idx}")



def load_model(name):
    if name is not None:
        model = A2CAgent(**post_process(dict(model_type="resnet", load_model=name)))
        model.eval()
    else:
        model = None

    return model


def run_one_game(env, model0, model1):
    print(env.info())

    data = json.loads(env.curr_json_str())

    print()

    if "state_display" in data:
        print("State display in json:")
        print(data["state_display"])

    print("Bidding #0")
    print(data["bidd"][0])

    print("Bidding #1")
    print(data["bidd"][1])

    precision = 5

    # Features from the evaluation.
    features = [ { k : torch.FloatTensor(v) for k, v in f.items() } for f in data["feature_history"] ]

    cnt = 0

    for tbl in range(2):
        print(f"Bidding table {tbl}")
        candidate_bidd = data["bidd"][tbl]["otherSeq"]
        curr_bidd = data["bidd"][tbl]["seq"]

        for i, (bid, candidate_bid) in enumerate(zip(curr_bidd, candidate_bidd)):
            f = env.feature()

            compare_tensor_dict(f, features[cnt])
            cnt += 1

            batch_unsqueeze(f)

            model = model0 if (i + tbl + data["dealer"]) % 2 == 0 else model1

            with torch.no_grad():
                reply = model.act_greedy(f)

            batch_squeeze(reply)

            action = reply["a"].item()
            prob = reply["pi"][action].item()

            print(f"   [{i}] Bid {bidIdx2str(action)}: {prob:.{precision}f}")
            print("          In replay:")
            for b in candidate_bid:
                print(f"              {b['bid']}: {b['prob']:.{precision}f}")
            print()

            action_ref = card_utils.bidStr2idx(candidate_bid[0]["bid"])

            env.step(action_ref)

        assert env.subgame_end(), "Subgame should end here!"

rank_checks = dict()
for rank in card_utils.IMPORTANT_RANKS:
    rank_checks[rank] = [
        card_utils.card2idx(s, rank) for s in card_utils.IDX2SUIT
    ]

suit_checks = OrderedDict()
for suit in card_utils.IDX2SUIT:
    suit_checks[suit] = [
        card_utils.card2idx(suit, rank) for rank in card_utils.IDX2RANK
    ]

def get_deals_from_uniform(true_cards, n, curr_seat=0):
    true_cards_absolute = [(v + curr_seat) % card_utils.NUM_PLAYERS
                           for v in true_cards.squeeze().tolist()]
    deal = DealWalk.from_card_map(true_cards_absolute)

    deals = []
    for i in range(n):
        deal.shuffle3(fix_player=curr_seat)
        deals.append(deepcopy(deal))

    return deals


def check_deals(belief_model,
                init_rep,
                true_cards,
                num_sample,
                curr_seat=0,
                stats=None):
    print(f"Seat {card_utils.IDX2SEATSE[curr_seat]} [{curr_seat}]: ")

    # Sample hands.
    with torch.no_grad():
        res = belief_model.sample(
            batch_dup(num_sample, init_rep),
            batch_dup(num_sample, true_cards)
        )

    cards = res["cards"]
    probs = res["probs"]
    orders = res["orders"]

    for i in range(num_sample):
        cards_loc, cards_list = analyze_sample(cards[i], probs[i], orders[i], true_cards, curr_seat, stats=stats)

        # plot_sample(cards_loc, cards_list)


def analyze_sample(cards, probs, orders, true_cards, seat, stats=None):
    cards = cards.tolist()
    probs = probs.tolist()
    orders = orders.tolist()
    true_cards = true_cards.squeeze().tolist()

    cards_loc = defaultdict(list)
    cards_list = []
    for idx, (c, o, p, t_c) in enumerate(zip(cards, orders, probs, true_cards)):
        if o == -1:
            continue
        other_seat_str = card_utils.IDX2SEATS[(seat + c) %
                                              card_utils.NUM_PLAYERS]
        card_str = card_utils.idx2card(idx)
        cards_loc[other_seat_str].append((card_str, p))

        if c == t_c:
            stats["accu"][idx] += 1

        if o >= 1:
            cards_list.append((o, card_str))

    cards_list = [ card_str for o, card_str in sorted(cards_list, key=lambda x: x[0]) ]

    # suit frequency.
    suit_len_diff = torch.zeros(card_utils.NUM_SUITS)

    for k, (suit, sel) in enumerate(suit_checks.items()):
        for i in range(card_utils.NUM_PLAYERS):
            suit_len = len([
                idx for idx, _seat in enumerate(cards)
                if _seat == i and idx in sel
            ])
            suit_len_true = len([
                idx for idx, _seat in enumerate(true_cards)
                if _seat == i and idx in sel
            ])
            suit_len_diff[k] += abs(suit_len - suit_len_true)
        suit_len_diff[k] /= card_utils.NUM_PLAYERS

    stats["suit_len_diff"] += suit_len_diff

    # check whether we have 13 cards for each player.
    hand_len_diff = torch.zeros(card_utils.NUM_PLAYERS)
    for _seat in cards:
        hand_len_diff[_seat] += 1

    stats["hand_len_diff"] += (hand_len_diff - 13).abs()
    stats["n"] += 1

    # import pdb
    # pdb.set_trace()

    return cards_loc, cards_list


def plot_sample(cards_loc, cards_list):
    print(f"  Sample {i}. Pred: {'-'.join(cards_list)}")
    for other_seat_str in card_utils.IDX2SEATS:
        s = ' '.join([
            f"{card_str} ({p:.2f})"
            for card_str, p in cards_loc[other_seat_str]
        ])
        print(f"    {other_seat_str}: {s}")


def run_belief(sampler, belief_model, stats=None):
    num_sample = 30

    for tbl in range(2):
        if sampler.set_tbl(tbl):
            continue

        # Run belief model
        for seat in range(card_utils.NUM_PLAYERS):
            sampler.set_seat(seat)
            f = sampler.f

            with torch.no_grad():
                init_rep = belief_model.get_init_rep(f)

            check_deals(belief_model, init_rep, f["cards"], num_sample, curr_seat=seat, stats=stats)


def compare_future_tricks(future_tricks_gt, avg_tricks):
    delta_tricks = avg_tricks - future_tricks_gt

    bias_tricks = np.sum(delta_tricks) / 13 
    ft_diff_per_card = np.linalg.norm(delta_tricks) / math.sqrt(13)

    # Also check whether the top-1 match is the same as the gt top-1 match.
    max_val = np.max(future_tricks_gt)
    max_indices_gt = np.where(future_tricks_gt == max_val)[0]

    max_val = np.max(avg_tricks)
    max_indices = np.where(avg_tricks == max_val)[0]

    ft_diff_opt_gt_set_per_card = np.linalg.norm(delta_tricks[max_indices_gt]) / math.sqrt(len(max_indices_gt))
    ft_diff_opt_set_per_card = np.linalg.norm(delta_tricks[max_indices]) / math.sqrt(len(max_indices))

    max_indices_gt = set(max_indices_gt)
    max_indices = set(max_indices)

    # Check IOU (not a good metric actually)
    intersect = len(max_indices_gt.intersection(max_indices))
    union = len(max_indices_gt.union(max_indices))
    iou = intersect / union

    # Whether max_indices is subset of max_indices_gt?
    # 1.0 if all max element of max_indices is inside max_indices_gt, which is good.
    subset_ratio = sum([ x in max_indices_gt for x in max_indices]) / len(max_indices)

    return dict(
        ft_diff_per_card = ft_diff_per_card,
        iou=iou,
        subset_ratio=subset_ratio,
        ft_diff_opt_gt_set_per_card=ft_diff_opt_gt_set_per_card,
        ft_diff_opt_set_per_card=ft_diff_opt_set_per_card,
        bias_tricks=bias_tricks
    )


def run_opening_lead_direct(sampler, belief_model, verbose=False):
    all_results = dict()
    all_results["dealer"] = sampler.data["dealer"]
    all_results["pbn"] = sampler.data["pbn"]

    for tbl in range(2):
        if not sampler.set_tbl(tbl):
            continue
        sampler.set_opening_lead()

        gt_info = sampler.get_gt_sample()
        future_tricks_gt = gt_info["future_tricks_gt"]

        with torch.no_grad():
            pred_tricks = belief_model.mean_tricks(sampler.f)

        # Given the predict, check whether they are the same.
        # pred = output["pred"].squeeze()
        # pred_tricks = pred.max(dim=1)[1]
        # pred_tricks[pred_tricks == 13 + 1] = -1

        pred_tricks.squeeze_()
        res = compare_future_tricks(future_tricks_gt, pred_tricks.cpu().numpy())
        all_results[f"stats.-direct.{tbl}"] = res
        all_results[f"stats.n.{tbl}"] = 1

        # import pdb
        # pdb.set_trace()

    return all_results


def run_opening_lead(args, 
                     sampler,
                     belief_model,
                     direct_model=None,
                     weight_model=None,
                     num_sample=100,
                     verbose=False):
    all_results = dict()
    all_results["dealer"] = sampler.data["dealer"]
    all_results["pbn"] = sampler.data["pbn"]

    all_samples = []

    for tbl in range(2):
        if not sampler.set_tbl(tbl):
            continue
        sampler.set_opening_lead()

        gt_info = sampler.get_gt_sample()
        future_tricks_gt = gt_info["future_tricks_gt"]

        all_results[f"ft-tricks-gt-{tbl}"] = future_tricks_gt
        all_results[f"strain-{tbl}"] = sampler.strain
        all_results[f"declarer-{tbl}"] = sampler.declarer
        all_results[f"seat-{tbl}"] = sampler.seat

        if belief_model is not None:
            ##### Posterior sampling
            samples = sampler.sample_deals(belief_model, num_sample)
            weights = sampler.get_weights(args, weight_model, samples)
            deals = samples["deals"]
            future_tricks, _ = get_future_tricks(deals, sampler.strain_idx, sampler.first)
            samples["future_tricks"] = future_tricks
            samples["gt_info"] = gt_info
            samples["f"] = sampler.f
            samples["bidd"] = sampler.curr_bidd
            # Relative dealer w.r.t the opening leader.
            samples["relative_dealer"] = (sampler.data["dealer"] + 4 - sampler.seat) % 4 

            # #print("From posterior")
            # #print_simulation(future_tricks_gt, future_tricks)
            weighted_avg_tricks = np.dot(weights, future_tricks)
            avg_tricks = np.mean(future_tricks, axis=0)
            # avg_tricks = get_deals_from_posterior2(belief_model, f, curr_seat=0)

            alpha = 0.1
            if direct_model is not None:
                prior_tricks = sampler.get_deals_from_prior(direct_model)
                samples["prior_tricks"] = prior_tricks
                
                if verbose:
                    print_tricks(prior_tricks, "prior_tricks")
                weighted_avg_tricks = (
                    1.0 - alpha) * prior_tricks + alpha * weighted_avg_tricks

            if verbose:
                print_tricks(weighted_avg_tricks, "weighted_avg_tricks")
                print_tricks(avg_tricks, "avg_tricks")
                print_tricks(future_tricks_gt, "gt_tricks")

            # gt_table = f["cards"].detach().cpu().numpy()
            # gt_table = np.reshape(gt_table, (4, 13))
            # print("gt_table = ")
            # print(f"{gt_table}")
            #
            # max_idx = np.where(weights == np.max(weights))[0]
            # for i in max_idx:
            #     mx_table = deals[i].card_map
            #     mx_table = np.reshape(mx_table, (4, 13))
            #     print("mx_table = ")
            #     print(f"{mx_table}")
            #     print(f"weight = {weights[i]}")
            #     print("mx_tricks = ")
            #     print(f"{future_tricks[i].reshape((4, 13))}")
            res = compare_future_tricks(future_tricks_gt, weighted_avg_tricks)
            res["success_rate"] = samples["success_rate"]
            all_results[f"avg-ft-tricks-posterior-{tbl}"] = weighted_avg_tricks
            all_results[f"stats.-posterior.{tbl}"] = res

            del samples["deals"]
            all_samples.append(samples)

        else:
            #### Uniform sampling
            deals = get_deals_from_uniform(sampler.f["cards"], num_sample, curr_seat=sampler.first)
            future_tricks, _ = get_future_tricks(deals, sampler.strain_idx, sampler.first)
            #print("From uniform")
            #print_simulation(future_tricks_gt, future_tricks)
            avg_tricks = np.mean(future_tricks, axis=0)
            res = compare_future_tricks(future_tricks_gt, avg_tricks)
            all_results[f"avg-ft-tricks-uniform-{tbl}"] = avg_tricks
            all_results[f"stats.-uniform.{tbl}"] = res

        all_results[f"stats.n.{tbl}"] = 1

    return all_results, all_samples

def replace_array_with_list(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, list):
        return [ replace_array_with_list(v) for v in d ]
    elif isinstance(d, dict):
        return { k : replace_array_with_list(v) for k, v in d.items() }
    else:
        return d

class Task(mp.Process):
    def __init__(self, task_idx, args, belief_model, direct_model,
                 weight_model, indices, counter_q):
        super().__init__()

        self.db, self.env = factory.create_single_env(args.db_filename, 0, 0, num_thread=1)
        self.sampler = BeliefSampler(self.env)

        self.indices = indices
        self.counter_q = counter_q
        self.task_idx = task_idx
        self.num_sample = args.num_sample
        self.belief_model = belief_model
        self.direct_model = direct_model
        self.weight_model = weight_model
        self.args = args
        if args.additional_info_file is not None:
            self.additional_info = torch.load(args.additional_info_file)
        else:
            self.additional_info = dict()

        self.all_samples = []

    def run_one(self, idx):
        # print(f"Case {i}, idx: {idx}")

        # run_one_game(data, env, model0, model1, belief_model)
        # Use dealer / vul within the dataset.
        # run_belief(self.sampler, belief_model, stats=stats)

        more_info = self.additional_info.get(idx, {})
        self.sampler.locate(idx)

        if self.args.belief_model_type == "direct":
            res = run_opening_lead_direct(self.sampler, self.belief_model)
        else:
            res, samples = run_opening_lead(self.args, self.sampler,  
                                   self.belief_model,
                                   self.direct_model,
                                   self.weight_model,
                                   num_sample=self.num_sample)

        self.stats = accu_stats(res)
        self.res = res
        self.all_samples += samples

    def run(self):
        f = open(f"output-{self.task_idx}", "w")

        for idx in self.indices:
            '''
            self.stats = defaultdict(lambda : 0)
            self.stats.update(dict(
                accu = np.zeros(card_utils.NUM_CARDS),
                suit_len_diff = np.zeros(4),
                hand_len_diff = np.zeros(4),
            ))
            '''
            self.run_one(idx)

            #res_stats = { k : v for k, v in res.items() if k.startswith("stats.") }
            #print(f"res: {res_stats}")
            #print(f"stats: {stats}")

            self.counter_q.put(self.stats)

            res = replace_array_with_list(self.res)
            res["idx"] = idx
            json.dump(res, f)
            f.write("\n")
            f.flush()

        f.close()
        torch.save(self.all_samples, f"samples-{self.task_idx}.torch")

import logging
log = logging.getLogger(__file__)

@hydra.main(config_path="conf/belief_analyze.yaml", strict=True)
def main(args):
    log.info(common_utils.init_context(args))
    log.info(f"Start time: {datetime.now()}")

    belief_model = create_belief_model(args.belief_model_type, args.belief_model_name)
    direct_model = create_direct_model(args.direct_model_name)
    weight_model = create_weight_model(args.weight_model_name)

    model0 = load_model(args.model0)
    model1 = load_model(args.model1)

    #import pdb
    #pdb.set_trace()

    db, env = factory.create_single_env(args.db_filename, 0, args.seed, num_thread=1)
    dataset_size = db.get_dataset_size()

    random.seed(args.seed)

    stats = None

    # Get a list
    choices = list(range(dataset_size))
    random.shuffle(choices)

    log.info(f"Start running: {datetime.now()}")

    if args.num_process > 1:
        n_per_task = args.n // args.num_process
        tasks = []
        counter_q = mp.Queue()

        for i in range(args.num_process):
            this_choice = choices[i*n_per_task:(i+1)*n_per_task]
            tasks.append(
                Task(i, args, belief_model, direct_model, weight_model,
                     this_choice, counter_q))

        for task in tasks:
            task.start()

        total = args.num_process * n_per_task
        with tqdm.tqdm(total=total) as pbar:
            for i in range(total):
                stats = add_stats(stats, counter_q.get())

                if args.display_freq > 0 and i % args.display_freq == 0:
                    log.info(get_stats(deepcopy(stats)))
                pbar.update(1)

        for task in tasks:
            task.join()
    else:
        # simple process for debug purpose.
        if args.indices is not None:
            this_choice = args.indices
        else:
            this_choice = choices[:args.n]

        task = Task(0, args, belief_model, direct_model, weight_model,
                    this_choice, None)
        for idx in tqdm.tqdm(this_choice):
            task.run_one(idx)
            stats = add_stats(stats, task.stats)


    log.info("Finally:")
    log.info(args.belief_model_name)
    log.info(args.belief_model_type)
    log.info(get_stats(deepcopy(stats)))
    log.info(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()
