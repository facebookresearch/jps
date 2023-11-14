import argparse
import os
import random
import readline
import sys
import time
import timeit

from copy import deepcopy

import card_utils
from sample_utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument("--init_pbn", type=str, default=None)

args = parser.parse_args()

dd_table_func = compute_dd_table_backup

while True:
    pbn = input("> ")
    pbn = pbn.strip()
    if pbn == "":
        continue

    if pbn == "q":
        break

    pbn = pbn.strip()

    if pbn.startswith("[Deal"):
        ws = [DealWalk(pbn)]
        dds_tables = dd_table_func(ws)

        for tbl in dds_tables:
            print_dd_table(tbl)

    elif pbn.startswith("PBN"):
        items = pbn.split(" ", 5)

        dealer = int(items[1])
        vul = int(items[2])
        # Here assume DDS strain convension.
        trump = card_utils.strain_swap(int(items[3]))
        first = int(items[4])
        pbn = "[Deal " + items[5] + "]"

        print(f"dealer: {dealer}, vul: {vul}, "
              f"strain: {card_utils.IDX2STRAIN[trump]}, first: {first}")

        deals = [DealWalk(pbn)]
        dds_tables = dd_table_func(deals)

        for tbl in dds_tables:
            print_dd_table(tbl)

        future_tricks_gt, _ = get_future_tricks(deals, trump, first)
        future_tricks_gt = future_tricks_gt[0]
        future_tricks_gt = future_tricks_gt.astype('i4')
        print_simulation(future_tricks_gt, future_tricks=None)

    elif pbn.startswith("SAMP"):
        items = pbn.split(" ")
        num_sample = int(items[1])

        last_deal = deals[0]
        possible_deals = []
        for i in range(num_sample):
            possible_deals.append(deepcopy(last_deal))
            possible_deals[-1].shuffle3(fix_player=first)
            print(f"{i}: {possible_deals[-1].get_pbn()}")

        future_tricks, _ = get_future_tricks(possible_deals, trump, first)
        print_simulation(future_tricks_gt, future_tricks)
