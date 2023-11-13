import sqlite3
import json
import argparse
import os
import sys
import glob
import yaml

sys.path.append("../dds")
import card_utils
# from card_utils import CardUtils, bidStr2idx, get_contract_declarer
from sample_utils import DealWalk, get_future_tricks, print_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--db', type=str, default="/checkpoint/yuandong/bridge/original/dda.db")
    parser.add_argument('--sec_idx', type=int, default=0)
    parser.add_argument('--num_sec', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--output_prefix', type=str, default="output")

    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    c = conn.cursor()

    c.execute(f"select count(*) from records")
    n = c.fetchone()[0]

    sec_len = int(n / args.num_sec)
    start = args.sec_idx * sec_len
    end = start + sec_len

    print(f"Interval: [{start}, {end})")

    f = open(args.output_prefix + "-" + str(args.sec_idx), "w")

    while start < end:
        this_end = min(start + args.batchsize, end)

        deals = []
        strains = []
        firsts = []
        inv_indices = []

        records = []
        for row in c.execute(f"select content from records where idx between {start} and {this_end-1}"):
            data = json.loads(row[0])
            records.append(data)

            curr_deal = DealWalk(data["pbn"])

            # Loop over all strains and all firsts.
            for strain_idx in range(card_utils.NUM_STRAINS):
                for first in range(card_utils.NUM_PLAYERS):
                    deals.append(curr_deal)
                    strains.append(strain_idx)
                    firsts.append(first)

                    # Point back to records
                    inv_indices.append(len(records) - 1)

        future_tricks, _ = get_future_tricks(deals, strains, firsts)
        future_tricks = future_tricks.astype('i4')

        for i, record_idx in enumerate(inv_indices):
            r = records[record_idx]
            if "fut" not in r:
                r["fut"] = [[None] * card_utils.NUM_CARDS
                            for strain_idx in range(card_utils.NUM_STRAINS)]
                r["card_map"] = deals[i].card_map

            fut = r["fut"][strains[i]]
            for k in range(card_utils.NUM_CARDS):
                if r["card_map"][k] == firsts[i]:
                    fut[k] = int(future_tricks[i][k])

        # Save it back
        for r in records:
            if args.debug:
                try:
                    for strain_idx in range(card_utils.NUM_STRAINS):
                        fut = r["fut"][strain_idx]

                        assert any(f is not None for f in fut), \
                            f"no entry in fut should be None: {fut}"

                        for first in range(card_utils.NUM_PLAYERS):
                            # PBN is dds format. so I need to swap the strain
                            print(f'PBN {r["dealer"]} {r["vul"]} '
                                  f'{card_utils.strain_swap(strain_idx)} '
                                  f'{first} {r["pbn"][6:-1]}')
                            this_fut = [
                                fut[k] if r["card_map"][k] == first else -1
                                for k in range(card_utils.NUM_CARDS)
                            ]
                            print_simulation(this_fut)
                except:
                    pass

                # For debug
                import pdb
                pdb.set_trace()

            f.write(json.dumps(r))
            f.write("\n")
        f.flush()

        start = this_end

    f.close()
