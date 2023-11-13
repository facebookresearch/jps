import glob
import hydra
import json
import math
import os
import sys
import torch

sys.path.append("../dds")
import card_utils
from sample_utils import *

import logging
log = logging.getLogger(__file__)

def extract_bid_seq(seq):
    bid_seq = list()
    last_pass_cnt = 3
    for bid_idx in seq:
        if bid_idx < 0:
            break

        if last_pass_cnt == 3:
            bid_seq.append(list())
            last_pass_cnt = 0

        bid_str = card_utils.bidIdx2str(bid_idx).strip()
        bid_seq[-1].append(bid_str)

        if bid_str == 'P' and len(bid_seq[-1]) > 1:
            last_pass_cnt += 1
        else:
            last_pass_cnt = 0

    assert len(bid_seq) == 2

    return bid_seq


@hydra.main(config_path="conf/check.yaml", strict=True)
def main(args):
    log.info(f"Working dir: {os.getcwd()}")
    log.info(args.pretty())

    filename = os.path.join(args.root, f"loss-batch-{args.idx}.pth")

    d = torch.load(filename)
    b = d["batch"] 
    batchsize = d["batch"]["s"].size(0)

    errs = torch.zeros(batchsize)
    infos = dict()

    for i in range(batchsize):
        idx = b["idx"][i].item()
        seat = b["seat"][i].item()
        strain = b["strain"][i].item()
        dealer = b["dealer"][i].item()
        vul = b["vul"][i].item()
        tbl = b["tbl"][i].item()
        fut = b["fut"][i]
        pred = d["res"]["pred"][i]
        s = b["s"][i]

        pred_tricks = pred.max(dim=1)[1]
        pred_tricks[pred_tricks == 14] = -1

        bid_seq = extract_bid_seq(b["bid"][i].tolist())

        card_map = (b["cards"][i] + seat) % 4
        deal = DealWalk.from_card_map(card_map)
        future_tricks, _ = get_future_tricks([deal], strain, seat) 

        # print(bid_seq[0])
        # print(bid_seq[1])

        additional_info = { 
            "idx": idx, 
            "tbl": tbl,
            "seat": seat,
            "strain": strain,
            "bidd" : [ dict(seq=bids) for bids in bid_seq ], 
            "dealer" : dealer, 
            "vul" : vul,
            "s": s,
            "fut": fut,
            "fut_recompute": future_tricks[0],
            "pred": pred,
            "pred_tricks": pred_tricks,
            "trick_l2_err": (pred_tricks - fut).float().norm().item() / math.sqrt(13)
        }

        infos[idx] = additional_info

        '''
        print(deal.get_pbn())

        print_simulation(future_tricks[0])

        print("Saved:")
        print_simulation(fut)

        #import pdb
        #pdb.set_trace()

        print("Predicted:")
        print_simulation(pred_tricks)

        print("Stored err: ")
        print(d["stats"]["trick_l2_err"])

        print("Computed err:")
        print(trick_l2_err)
        '''

        errs[i] = additional_info["trick_l2_err"]

    # print(sum(errs) / len(errs))
    min_err, min_idx = errs.min().item(), errs.argmin().item()
    max_err, max_idx = errs.max().item(), errs.argmax().item()

    min_g_idx = d["batch"]["idx"][min_idx]
    max_g_idx = d["batch"]["idx"][max_idx]

    print("Min err", min_err, min_idx, min_g_idx) 
    print("Max err", max_err, max_idx, max_g_idx) 
    print("avg: ", errs.mean().item())
    print(d["stats"])

    print(f"Saving to {args.output}")
    torch.save(infos, args.output) 

if __name__ == "__main__":
    main()
