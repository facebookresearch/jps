import argparse
import json

from collections import defaultdict, Counter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--first_n', type=int, default=None)

    args = parser.parse_args()

    records = defaultdict(lambda : defaultdict(list))

    i = 0

    for line in open(args.filename, "r"):
        if line.startswith("json_state:"):
            json_str = line[11:]
            data = json.loads(json_str)

            records[data["dataset_filename"]][data["data_idx"]].append(i)
            i += 1


    for k, v in records.items():
        print(f"{k}: {len(v)}")
        for kk, c in v.items():
            if len(c) != 1:
                print(f"   {kk}: {c}")
        
