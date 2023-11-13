import json
import numpy as np

filename = "/private/home/yuandong/bridge/python/outputs/2020-03-29/00-08-09/output-0"

ratios = []

for line in open(filename, "r"):
    data = json.loads(line)
    for i in range(2):
        probs = np.array(data["bidd"][i]["probs"])
        probs_alt = np.array(data["bidd"][i]["prob_alt"])

        trickTaken = data["bidd"][i]["trickTaken"] 
        trickTaken = data["bidd"][i]["trickTaken"] 

        log_likelihood = np.sum(np.log(probs))
        log_likelihood_alt = np.sum(np.log(probs_alt))
        log_likelihood_ratio = log_likelihood - log_likelihood_alt  

        ratios.append(log_likelihood_ratio)

ratios = np.array(ratios)
print(f"min: {np.min(ratios)}")
print(f"max: {np.max(ratios)}")
print(f"mean: {np.mean(ratios)}")

print(f"# < 0: {np.where(ratios < 0)[0].shape[0]} / {ratios.shape[0]}")
