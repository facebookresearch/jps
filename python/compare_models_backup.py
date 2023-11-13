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

baselines = [
        (0.12938958406448364, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/31', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=6,agent.params.explore_ratio=0.0,trainer.params.actor_sync_freq=25,githash=12e9816', 237),
        (0.12748542428016663, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/27', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=5,agent.params.explore_ratio=0.000625,trainer.params.actor_sync_freq=25,githash=12e9816', 239),
        (0.1273270845413208, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/33', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=6,agent.params.explore_ratio=0.000625,trainer.params.actor_sync_freq=25,githash=12e9816', 254),
        (0.12666457891464233, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/37', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=7,agent.params.explore_ratio=0.0,trainer.params.actor_sync_freq=25,githash=12e9816', 242),
        (0.1264958381652832, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/0', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=1,agent.params.explore_ratio=0.0,trainer.params.actor_sync_freq=12,githash=12e9816', 215),
        (0.12642917037010193, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/89', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0002,seed=5,agent.params.explore_ratio=0.00125,trainer.params.actor_sync_freq=25,githash=12e9816', 155),
        (0.12631458044052124, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/5', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=1,agent.params.explore_ratio=0.00125,trainer.params.actor_sync_freq=25,githash=12e9816', 233),
        (0.12628959119319916, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/40', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=7,agent.params.explore_ratio=0.00125,trainer.params.actor_sync_freq=12,githash=12e9816', 152),
        (0.1261499971151352, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/58', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=10,agent.params.explore_ratio=0.00125,trainer.params.actor_sync_freq=12,githash=12e9816', 99),
        (0.12612499296665192, '/checkpoint/yuandong/outputs/2020-01-16/21-27-19/3', 'trainer=selfplay,game=bridge,method=a2c,num_thread=25,num_game_per_thread=50,num_eval_per_thread=800,lr=0.0001,seed=1,agent.params.explore_ratio=0.000625,trainer.params.actor_sync_freq=25,githash=12e9816', 218)
]

search_models = {
    "nosearch": "/checkpoint/qucheng/outputs/2020-02-02/23-04-51/0/agent-64.pth",
    "1-seach_move_choice": "/checkpoint/qucheng/outputs/2020-02-04/11-39-05/7/agent-87.pth",
    "1-search_game_choice": "/checkpoint/qucheng/outputs/2020-02-04/13-36-37/1/agent-81.pth",
    "2-search_game_choice": "/checkpoint/qucheng/outputs/2020-02-02/23-04-51/11/agent-63.pth",
    "2-search_move_choice": "/checkpoint/qucheng/outputs/2020-02-03/12-35-52/8/agent-169.pth",
    "5pct": "/checkpoint/qucheng/outputs/2020-02-02/23-04-51/5/agent-250.pth",
    "10pct": "/checkpoint/qucheng/outputs/2020-02-02/22-55-22/5/agent-168.pth",
    "20pct": "/checkpoint/qucheng/outputs/2020-02-02/22-55-22/8/agent-99.pth",
}

b_models = [ "/checkpoint/yuandong/outputs/2020-01-03/13-54-29/16/agent-186.pth" ] + [ os.path.join(b[1], f"agent-{b[3]}.pth") for b in baselines ]

for b_model in b_models:
    for key, s in search_models.items():
        command = f"python main2.py num_thread=200 num_eval_per_thread=500 seed=1 method=a2c eval_only=true game=bridge agent.params.load_model={s} baseline=a2c baseline.agent.params.load_model={b_model}" 
        print(f"{key} versus {b_model}")
        print(command)
        print(run(command))

