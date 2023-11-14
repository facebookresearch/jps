# vim: set fileencoding=utf-8
import argparse
import os
import random
import sys
import time
import timeit

from copy import deepcopy

import card_utils
from sample_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--init_pbn", type=str, default=None)
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--main_player", type=int, default=0)

    args = parser.parse_args()

    deal = DealWalk(args.init_pbn)

    print("Current deal:")
    print(deal.get_pbn())

    # Opening lead now.
    first = args.main_player

    # The first one won't shuffle.
    rand_deal = deepcopy(deal)

    deals = []
    deals.append(deepcopy(rand_deal))

    print("Other deals:")
    for i in range(args.num_simulations):
        rand_deal.shuffle3(fix_player=args.main_player)
        print(rand_deal.get_pbn())
        deals.append(deepcopy(rand_deal))

    for strain_idx in range(card_utils.NUM_STRAINS):
        # future_tricks: [num_sim, 52]
        # mean_tricks: [num_sim]

        future_tricks, mean_tricks = get_future_tricks(deals, strain_idx,
                                                       first)

        print(f"Trump: {card_utils.IDX2STRAIN[strain_idx]}")
        print_simulation(future_tricks[0, :], future_tricks[1:, :])
        print()
