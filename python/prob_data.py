import pickle
import readline
import argparse
import random
import re
import os
import sys
import socket
import json
import io
import torch

from belief import BeliefDataset, BeliefDatasetWrapper

sys.path.append("../dds")
from sample_utils import DealWalk

import glob

class Prober:
    def __init__(self, prefixes):
        self.records = dict()
        self.freqs = dict()

        for prefix in prefixes.split(","):
            record_names = list(glob.glob(prefix + ".record"))
            idx_names = list(glob.glob(prefix + ".idx"))

            for record_name, idx_name in zip(record_names, idx_names):
                key = os.path.basename(record_name[:(len(record_name) - len(".record"))])

                print(f"Loading {record_name} and {idx_name}")
                self.records[key] = pickle.load(open(record_name, "rb"))
                self.freqs[key] = pickle.load(open(idx_name, "rb"))
                print(f"#record: {len(self.records[key])}, #idx: {len(self.freqs[key])}")

    def query(self, q):
        items = q.split(" ")
        record = self.records[items[0]]
        freq = self.freqs[items[0]]
        items = items[1:]

        ret = dict(stats=None, records=[])

        if items[0] == "new_situation":
            if len(items) == 2:
                idx = min(int(items[1]), len(record) - 1)
                idx = max(idx, 0)
            else:
                idx = random.randint(0, len(record) - 1)
            ret["records"] = [record[idx]]

        elif items[0] == "bid_seq":
            key = items[1]
            if key in freq:
                ret["records"] = [ record[idx] for idx in freq[key]["indices"] ]
                ret["stats"] = freq[key]["stats"]
            else:
                ret["stats"] = "No record found"
            print(f"{len(ret['records'])} Record found.")

        else:
            ret["stats"] = "Unknown!"

        return ret

    @classmethod
    def convert(cls, pkg):
        states = []
        for s in pkg["records"]:
            states.append(dict(state=DealWalk.from_pbn(s["state"])))
        return states, pkg["stats"]


class ProberBeliefSample:
    def __init__(self, root):
        dataset = BeliefDataset(root)
        d = BeliefDatasetWrapper(dataset.data, output_d=True)
        self.dataset = dataset
        self.d = d

    def query(self, q):
        i = int(q)
        _, ft_gt, cards, ft, prior_tricks, d = self.d[i]
        return dict(ft_gt=ft_gt, cards=cards, ft=ft, prior_tricks=prior_tricks,
                bidd=d["bidd"], dealer=d["relative_dealer"])

    @classmethod
    def convert(cls, pkg):
        states = []
        n = pkg["cards"].size(0)
        bidd = pkg["bidd"]
        dealer = pkg["dealer"]
        for i in range(n):
            deal = DealWalk.from_card_map(pkg["cards"][i].tolist())
            states.append(dict(state=deal, bidd=[dict(seq=bidd)], dealer=dealer))

        return states, None

class Client:
    def __init__(self, port=9999):
        self.port = port
        self.host = "localhost"

    def query(self, data):
        # Create a socket (SOCK_STREAM means a TCP socket)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Connect to server and send data
            sock.connect((self.host, self.port))

            sock.sendall(bytes(data + "\n", "utf-8"))

            # Receive data from the server and shut down
            received = b""
            while True:
                tmp = sock.recv(4096)
                if tmp is None or len(tmp) == 0:
                    break
                received += tmp

            buffer = io.BytesIO(received)
            return torch.load(buffer)

if __name__ == "__main__":
    ''' Test the client '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--server_port", type=int, default=None)

    args = parser.parse_args()

    if args.server_port is None:
        client = Prober(args.prefix)
    else:
        client = Client(port=args.server_port)

    while True:
        q = input("> ")
        ret = client.query(q)
        print(ret["stats"])
        # Enter debug model, until you press c

        import pdb
        pdb.set_trace()

