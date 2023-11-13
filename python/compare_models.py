import sys
import os
import re
import json
import pickle
import argparse
import pandas as pd

from pretrained_models import * 

cmd_template = "init; bridge; run_local \"main2.py num_thread=200 seed={seed} method=a2c eval_only=true game=bridge agent.params.load_model={model} baseline=a2c baseline.agent.params.load_model={b_model} use_search=false game.params.old_sampler={old_sampler} baseline.agent.params.explore_ratio={explore_ratio} agent.params.explore_ratio={explore_ratio} baseline.actor_gen.params.use_sampling_in_eval={use_sampling_in_eval} actor_gen.params.use_sampling_in_eval={use_sampling_in_eval}\""

matcher = re.compile(r"eval_score(.*?)avg:\s+([\-\d\.]+)")

models_tbl = {
    "baseline": baseline_models,
    "search": search_models,
    "search_Mar_3": search_models_Mar_3_traj,  
    "all": all_models
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_prefix', type=str, default=None)
    parser.add_argument('--agent_models', type=str, default="search")
    parser.add_argument('--baseline_models', type=str, default="baseline")

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--explore_ratio', type=float, default=0.0)
    parser.add_argument('--use_sampling_in_eval', action="store_true")
    parser.add_argument('--old_sampler', action="store_true")
    parser.add_argument('--first_n', type=int, default=None)
    
    args = parser.parse_args()

    records = []
    outputs = []
    cross_table = []

    f = open(args.output_prefix + ".txt", "w")

    models_agent = models_tbl[args.agent_models] 
    models_baseline = models_tbl[args.baseline_models] 

    for k, (b_key, b_model) in enumerate(models_baseline.items()):

        cross_table_row = dict()

        for j, (key, model) in enumerate(models_agent.items()):
            command = cmd_template.format(model=model,b_model=b_model,**args.__dict__)
            print(f"{key} versus {b_key}: {b_model}")
            print(command)

            result = run(["/bin/bash", "-i", "-c", command], shell=False)
            lines = result.split("\n")

            for i in range(5):
                l = lines[-i]
                if l.strip() == "":
                    continue

                m = matcher.search(l)
                if m is not None:
                    score = float(m.group(2)) 

                    entry = dict(baseline=b_key,baseline_filename=b_model,model=key,model_filename=model,score=score) 
                    print(entry)
                    records.append(entry)
                    outputs.append(result)

                    f.write(result)
                    f.flush()

                    cross_table_row[j] = score    

                    break

            if args.first_n is not None and j == args.first_n - 1:
                break

        cross_table.append(cross_table_row)

        if args.first_n is not None and k == args.first_n - 1:
            break

    print("===== Summary =====")
    print("Rows / Baselines:")
    for i, (k, v) in enumerate(models_baseline.items()):
        print(f"{i}: {k}")

    print()
    print("Columns / Agents:")
    for i, (k, v) in enumerate(models_agent.items()):
        print(f"{i}: {k}")

    df_records = pd.DataFrame(records)
    df_cross_table = pd.DataFrame(cross_table)

    print()
    print("Cross table (Agent over baseline)")
    print(df_cross_table)
            
    data = dict(records=df_records,outputs=outputs,args=args, cross_table=df_cross_table) 
    pickle.dump(data, open(args.output_prefix + ".pkl", "wb"))
