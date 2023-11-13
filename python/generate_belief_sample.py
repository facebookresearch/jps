import sys
sys.path.append("..")

import os

import torch
import numpy as np

sys.path.append("../dds")
import card_utils
from sample_utils import DealWalk, get_future_tricks, print_simulation, compute_dd_table_backup, print_bidding_table
from pybridge import factory
import json

import hydra

from utils_belief import *

import logging
log = logging.getLogger(__file__)

@hydra.main(config_path="conf/generate_belief_sample.yaml", strict=True)
def main(args):
    log.info(f"Working dir: {os.getcwd()}")
    log.info(args.pretty())

    belief_model = create_belief_model(args.belief_model_type, args.belief_model_name)

    db, env = factory.create_single_env(args.db_filename, 0, 0, num_thread=1)

    sampler = BeliefSampler(env)
    sampler.locate(args.idx)

    if "state_display" in sampler.data:
        print(sampler.data["state_display"])

    if "pbn" in sampler.data:
        print(sampler.data["pbn"])

    print(f"Dealer: {sampler.data['dealer']}")

    for tbl in range(2):
        if not sampler.set_tbl(tbl):
            continue

        sampler.set_opening_lead()
        print(f"=== Bidding table {tbl} ===")

        print(f"Bidd seq: {' '.join(sampler.curr_bidd)}")
        print_bidding_table(sampler.data["dealer"], sampler.curr_bidd)

        # Run belief model
        # for seat in range(card_utils.NUM_PLAYERS):
        print(f"Seat: {sampler.seat}")

        gt_info = sampler.get_gt_sample()
        future_trick_gt = gt_info["future_tricks_gt"]

        deal_gt = gt_info["deal"]
        dds_gt = compute_dd_table_backup(deal_gt)[0]

        for k in range(args.num_repeat):
            # Sample hands.
            samples = sampler.sample_deals(belief_model, args.num_sample)
            deals = samples["deals"]

            dds = compute_dd_table_backup(deals)
            future_tricks, _ = get_future_tricks(deals, sampler.strain_idx, sampler.first)
            logprobs = samples["probs"].log().sum(dim=1)

            probs = logprobs - logprobs.mean()
            probs = probs.exp()
            probs = probs / probs.sum()

            _, indices = torch.sort(probs, descending=True)
            # Note that for opening lead, we always check 3 (declarer is to the left of the current player, which is at 0)
            print(f"Gt DDS: {dds_gt[3,:].astype(int).tolist()}")

            probs = probs[indices]
            logprobs = logprobs[indices]
            dds = dds[indices, :, :]
            deals = [ deals[i] for i in indices ]
            future_tricks = future_tricks[indices, :]

            for i in range(args.num_sample):
                # print them out..
                # Note that for opening lead, we always check 3 (declarer is to the left of the current player)
                score = dds[i, 3, :].astype(int).tolist()
                pbn = card_utils.IDX2SEATS[sampler.seat] + ':' + deals[i].get_pbn_noprefix()
                print(f"{card_utils.rotate_nfirst(pbn)}   prob: {probs[i]:.4f}  raw_log_prob: {logprobs[i]:.4f}  dds: {score}")

            print(f"Mean DDS: {np.mean(dds[:, 3, :], axis=0)}")

            print_simulation(future_trick_gt, future_tricks=future_tricks)

if __name__ == "__main__":
    main()
