import sqlite3
import json
import argparse
import os
import sys
import glob
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--files', type=str, default="/checkpoint/qucheng/bridge/random_data/dda*")
    parser.add_argument('--sort_key', type=str, default=None, help="sort according to some key")
    parser.add_argument('--json_per_line', action="store_true")
    parser.add_argument('--transpose_ddt', action="store_true")
    parser.add_argument('--exclude_log', action="store_true")
    parser.add_argument('--exclude_hydra', action="store_true")
    parser.add_argument('--meta_other_info', type=str, default=None)
    parser.add_argument('--db', type=str, default="dda.db")

    args = parser.parse_args()

    if os.path.exists(args.db):
        print(f"Database {args.db} exists!")
        sys.exit(0)

    records = []

    for filename in glob.glob(args.files):
        if os.path.isdir(filename) or filename.endswith(".sh") or filename.endswith(".py"): 
            continue

        print(filename)

        if args.json_per_line:
            for line in open(filename):
                data = json.loads(line)
                if args.transpose_ddt:
                    # Transpose the 4 x 5 table from [row1, row2, row3, row4] to [col1 .. col5]
                    ddt = []
                    for j in range(5):
                        for i in range(4):
                            ddt.append(data["ddt"][5 * i + j])
                    data["ddt"] = ddt
                records.append(data)
        else:
            deal_read = False
            entry = dict()
            for line in open(filename):
                if not deal_read:
                    entry["pbn"] = line.strip()
                    deal_read = True
                else:
                    entry["ddt"] = [ int(v) for v in line.strip().split(" ") ]
                    deal_read = False
                    records.append(entry)
                    entry = dict()

    if args.sort_key is not None:
        records = sorted(records, key=lambda x: x[args.sort_key])

    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    c.execute("create table records (IDX INTEGER PRIMARY KEY, CONTENT TEXT)")

    for cnt, entry in enumerate(records):
        c.execute(f"insert into records values ({cnt}, '{json.dumps(entry)}')")

    c.execute("create table meta (name VARCHAR, content TEXT)")
    for k, v in args.__dict__.items():
        c.execute(f"insert into meta values('{k}', '{v}')") 

    # Save all logs if there is any. 
    root = os.path.dirname(args.files) 
    if not args.exclude_log:
        for logfile in glob.glob(os.path.join(root, "*.log")):
            data = json.dumps(open(logfile).readlines())
            c.execute(f"insert into meta values('{logfile}', '{data}')") 

    # Save all hydra settings.
    if not args.exclude_hydra:
        if os.path.exists(os.path.join(root, ".hydra")):
            for config_file in glob.glob(os.path.join(root, ".hydra/*.yaml")):
                if config_file.endswith("hydra.yaml"): 
                    continue
                data = json.dumps(yaml.load(open(config_file)))
                c.execute(f"insert into meta values('{config_file}', '{data}')") 

    if args.meta_other_info is not None:
        c.execute(f"insert into meta values('other_info', '{args.meta_other_info}')") 
        
    conn.commit()
    conn.close()

    print(f"Total records: {len(records)}. Saved to {args.db}")
                
            
            



