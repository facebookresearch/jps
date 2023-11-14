# vim: set fileencoding=utf-8
import time
import argparse
import timeit
import sys
import os
import random

from sample_utils import *

if __name__ == "__main__":
    '''
    python sample2.py --n 10 --i 0
    
    it will save to data-0.txt. Sample results:
        [Deal "N:KQJ3.KQ7.6.J9872 87642.842.KQJ.Q6 AT.T65.A9874.AT3 95.AJ93.T532.K54"]
        11 2 11 2 8 5 8 5 7 6 7 6 8 4 8 4 8 4 8 4
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--i", type=int, help="Index")
    parser.add_argument("--init_pbn", type=str, default=None)
    parser.add_argument("--n", type=int, default=40000)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--action_type", type=str, choices=["swap2cards", "shuffle3", "swap2cards3"], default="swap2cards")
    parser.add_argument("--prefix", type=str, default="data")

    args = parser.parse_args()

    random.seed(args.i)

    # n = 100
    # print(timeit.timeit(sample, number=n) / n)

    dd_table_func = compute_dd_table_backup

    start = time.perf_counter() 

    with open(f"{args.prefix}-{args.i}.txt", "w") as f:
        ws = [ DealWalk(args.init_pbn) for i in range(args.n) ]

        for k in range(args.step):
            dds_tables = dd_table_func(ws)

            for i in range(args.n):
                s = "[Deal \"" + ws[i].get_pbn() + "\"]\n" 
                tricks = [ str(trick) for trick in np.reshape(dds_tables[i, :, :].T, 20).astype('i4') ]
                s += " ".join(tricks) + "\n" 

                f.write(s)
                if args.n == 1:
                    print(s)

            f.flush()
                
            for w in ws:
                getattr(w, args.action_type)()

    print("Time spent: ", time.perf_counter() - start)

