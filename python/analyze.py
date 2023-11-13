import os
import sys

from pretrained_models import * 

cmd_template = "python main2.py num_thread=200 seed=78924 eval_only=true method=a2c agent.params.load_model={model} baseline=a2c baseline.agent.params.load_model={model} game=bridge game.params.display_freq=1 > {output}.bid" 

for key, model in all_models.items():
    cmd = cmd_template.format(model=model, output=key)
    print(cmd)
    run(cmd)

