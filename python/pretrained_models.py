import os
import sys
from collections import OrderedDict

a2c_models = OrderedDict({
    "basic": "/checkpoint/yuandong/outputs/2020-01-03/13-54-29/16/agent-186.pth",
})

search_models = OrderedDict({
    "nosearch": "/checkpoint/qucheng/outputs/2020-02-02/23-04-51/0/agent-64.pth",
    "nosearch355": "/checkpoint/qucheng/outputs/2020-02-02/23-04-51/0/agent-355.pth",
    "1-seach_thread_choice": "/checkpoint/qucheng/outputs/2020-02-04/11-39-05/7/agent-87.pth",
    "1-search_move_choice": "/checkpoint/qucheng/outputs/2020-02-04/13-36-37/1/agent-81.pth",
    "2-search_thread_choice": "/checkpoint/qucheng/outputs/2020-02-02/23-04-51/11/agent-63.pth",
    "2-search_move_choice": "/checkpoint/qucheng/outputs/2020-02-03/12-35-52/8/agent-169.pth",
    "2-search-5pct": "/checkpoint/qucheng/outputs/2020-02-02/23-04-51/5/agent-250.pth",
    "2-search-10pct": "/checkpoint/qucheng/outputs/2020-02-02/22-55-22/5/agent-168.pth",
    "2-search-20pct": "/checkpoint/qucheng/outputs/2020-02-02/22-55-22/8/agent-99.pth",
    "2-search-model-14-days": "/private/home/yuandong/bridge/python/outputs/2020-02-10/07-16-11/agent-1610.pth",
    "2-search-model-14-days-2": "/private/home/yuandong/bridge/python/outputs/2020-02-10/07-16-11/agent-1712.pth",
    "2-search-model-14-days-3": "/private/home/yuandong/bridge/python/outputs/2020-02-10/07-16-11/agent-1723.pth"
})

model_root = "/checkpoint/yuandong/bridge/models/"

search_models_Mar_3_traj = OrderedDict({
    "Mar3_644_0531": os.path.join(model_root, "agent-644.pth"),
    "Mar3_1442_0537": os.path.join(model_root, "agent-1442.pth"),
    "Mar3_1691_0547": os.path.join(model_root, "agent-1691.pth"),
    "Mar3_1839_0567": os.path.join(model_root, "agent-1839.pth"),
    "Mar3_2090_0540": os.path.join(model_root, "agent-2090.pth"),
    "Mar3_2218_0554": os.path.join(model_root, "agent-2218.pth"),
    "Mar3_2296_0587": os.path.join(model_root, "agent-2296.pth"),
    "Mar3_2333_0503": os.path.join(model_root, "agent-2333.pth"),
    "Mar3_2351_0549": os.path.join(model_root, "agent-2351.pth"),
})

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

baseline_models = OrderedDict()
baseline_models["baseline"] = "/checkpoint/yuandong/outputs/2020-01-03/13-54-29/16/agent-186.pth"
for i, b in enumerate(baselines):
    baseline_models[str(i)] = os.path.join(b[1], f"agent-{b[3]}.pth")

all_models = OrderedDict()
for models, prefix in zip( (a2c_models, search_models, baseline_models), ("a2c", "search", "baseline") ):
    for key, model in models.items():
        all_models[prefix + "." + key] =  model


import subprocess
def run(command, shell=True):
    try:
        result = subprocess.check_output(command, shell=shell) 
    except subprocess.CalledProcessError as e:
        result = e.output
    except KeyboardInterrupt:
        print('Keyboard interrupt!')
        sys.exit(0)

    return result.decode("utf-8")
