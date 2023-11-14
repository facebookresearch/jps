# vim: set fileencoding=utf-8
from collections import Counter
import time
import argparse
import timeit
import pickle

from sample_utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument("--i", type=int, help="Index")
parser.add_argument("--save_file", type=str, default="tmp.pkl")
parser.add_argument("--load", action="store_true")

args = parser.parse_args()

def sample():
    deal = dealer()
    dds_table = deal.dd_table()
    print(f"{deal}: dds: {dds_table}")

    '''
    dds_table2 = get_dds_table(deal)
    res = compare(dds_table, dds_table2)
    print(f"{deal}: dds: {dds_table}, dds2: {dds_table2}, matched: {res}")
    if not res:
        print("Not matched!! Something wrong!")
        raise RuntimeError
    '''

# n = 100
# print(timeit.timeit(sample, number=n) / n)

bs = 100
step = 5

if not args.load:

    dd_table_func = compute_dd_table_backup

    start = time.perf_counter() 

    # init_deal = [ [ "AT7652", "6", "AKJ", "K85" ], ["QJ", "9754", "T42", "A643" ], [ "K83", "AK83", "", "QJT972" ], ["94", "QJT2", "Q987653", ""] ]

    ws = [ DealWalk() for i in range(bs) ]

    situation0 = str(ws[0])

    dds_table0 = np.squeeze(dd_table_func([ ws[0] ]))
    #dds_table0_backup = np.squeeze(compute_dd_table_backup([ ws[0] ]))
    #print(np.linalg.norm(dds_table0 - dds_table0_backup))

    situations = [ [ None for i in range(bs) ] for j in range(step) ]
    dds_tables = np.zeros((step, bs, 4, 5))

    for i in range(step):
        for w in ws:
            w.step()

        dds_tables[i, :, :, :] = dd_table_func(ws)
        #dds_tables_backup = compute_dd_table_backup(ws)
        #print(np.linalg.norm(dds_tables[i, :, :, :] - dds_tables_backup))

        for k, w in enumerate(ws):
            situations[i][k] = str(w)

    diffs = np.sum(np.abs(dds_tables - dds_table0[None, None, :, :]), axis=(2,3))

    print("Time spent: ", time.perf_counter() - start)

    results = dict(situation0=situation0, dds_table0=dds_table0, diffs=diffs, situations=situations, dds_tables=dds_tables)
    pickle.dump(results, open(args.save_file, "wb"))

else:
    results = pickle.load(open(args.save_file, "rb"))

diffs = results["diffs"]
situations = results["situations"]
dds_tables = results["dds_tables"]

situation0 = results["situation0"]
dds_table0 = results["dds_table0"]

print(f"mean: {np.mean(diffs, axis=1)}, max: {np.max(diffs, axis=1)}, min: {np.min(diffs, axis=1)}")

for k in range(step):
    print(f"#move: {k + 1}: ")
    indices = np.argsort(diffs[k,:])

    for idx in indices[-5:]:
        print(f"{diffs[k,idx]}")
        print(f"{situations[k][idx]}")
        print(f"{situation0}")

        print(f"{dds_tables[k][idx]}")
        print(f"{dds_tables[k][idx] - dds_table0}")


