import sys
import os
import json
import subprocess

def run(command):
    try:
        result = subprocess.check_output(command, shell=True) 
    except subprocess.CalledProcessError as e:
        result = e.output
    return result.decode("utf-8")


records = json.load(open(sys.argv[1], "r"))

baseline = "/checkpoint/yuandong/outputs/2020-01-03/13-54-29/16/agent-186.pth" 

for record in records[::-1]:
    print(record)

    hid_dim = 200
    for item in record[2].split(","):
        k, v = item.split("=")
        if k.find("hid_dim") >= 0:
            hid_dim = int(v)

    filename = os.path.join(record[1], f"agent-{record[3]}.pth")

    print(f"Test: {filename}")
    for seed in range(5):
        command = f"python main2.py num_thread=200 seed={seed + 1} method=a2c eval_only=true game=bridge agent.params.load_model={filename} agent.params.hid_dim={hid_dim} baseline=baseline16" 
        print(f"Seed: {seed + 1}")
        print(command)

        print(run(command))

        command = f"python main2.py num_thread=200 seed={seed + 1} method=a2c eval_only=true game=bridge agent.params.load_model={filename} agent.params.hid_dim={hid_dim} baseline=a2c baseline.agent.params.load_model={baseline}"

        print(run(command))



