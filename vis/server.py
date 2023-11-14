import sys
import os

import socketserver
import re
import random
import json
import pickle
import io
import importlib
import torch
import argparse

sys.path.append("../python")

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """
    def handle(self):
        # self.request is the TCP socket connected to the client
        data = self.request.recv(1024).decode('utf-8').strip()
        print("{} wrote:".format(self.client_address[0]))
        print(f"Received: {data}")

        global prober
        ret = prober.query(data)
        ret["_clsname"] = prober.__class__.__name__

        b = io.BytesIO()
        s = torch.save(ret, b)
        self.request.sendall(b.getbuffer())


if __name__ == "__main__":
    HOST, PORT = "localhost", 9999

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--prober_type", type=str)
    parser.add_argument("--prefix", type=str)

    args = parser.parse_args()

    global prober
    prob_module = importlib.import_module("prob_data")
    prober = getattr(prob_module, args.prober_type)(args.prefix)

    # Create the server, binding to localhost on port 9999
    with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
        # Activate the server; this will keep running until you
        # interrupt the program with Ctrl-C
        server.serve_forever()
