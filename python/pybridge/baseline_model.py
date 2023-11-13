import torch
import os
from torch import nn
import scipy
import scipy.io as sio
import numpy as np
from typing import Tuple, Dict, List
try:
    from typing_extensions import Final
except:
    # If you don't have `typing_extensions` installed, you can use a
    # polyfill from `torch.jit`.
    from torch.jit import Final

class BaselineBridgeModel(torch.jit.ScriptModule):
    __constants__ = ["num_action", "hidden_dim", "input_dim", "hand_dim", "bid_index"]

    def __init__(self):
        linears: List[nn.Module]
        super().__init__()

        curr_dir = os.path.dirname(os.path.abspath(__file__))

        roots = [
             os.path.join(curr_dir, "../../Automatic-Bridge-Bidding-by-Deep-Reinforcement-Learning"),
             "/home/yuandong/baselines"
        ]

        model_name = "model_valcost_1.063088e-01_totalbid4_4_128_50_3.204176e-03_9.800000e-01_8.200000e-01_alpha1.000000e-01.mat"

        mat_content = None
        for root in roots:
            filename = os.path.join(root, model_name)
            if os.path.exists(filename):
                mat_content = sio.loadmat(filename)
                break

        if mat_content is None:
            raise RuntimeError(r"Cannot find {model_name}!")

        self.num_action = 36
        self.hidden_dim = 128
        self.input_dim = 94
        self.hand_dim = 52
        self.bid_index = 93
        #self.linears = [None] * 20
        self.linears = torch.nn.ModuleList([nn.Linear(1, 1) for i in range(20)])
        self.leaky_relu = nn.LeakyReLU(0.2)

        for i in range(4):
            for j in range(5):
                if j == 0:
                    if i == 0:
                        in_channel = self.hand_dim
                    else:
                        in_channel = self.bid_index
                else:
                    in_channel = self.hidden_dim
                if j == 4:
                    out_channel = self.num_action
                else:
                    out_channel = self.hidden_dim
                    
                self.linears[i * 5 + j] = nn.Linear(in_channel, out_channel)
                self.linears[i * 5 + j].weight = torch.nn.Parameter(torch.from_numpy(mat_content["WW_qlearning"][0][i][0][j]).cuda())
                self.linears[i * 5 + j].bias = torch.nn.Parameter(torch.from_numpy(mat_content["BB_qlearning"][0][i][0][j]).squeeze().cuda())
    
    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = obs["baseline_s"].cuda()

        bs = x.size(0)
        #assert((x[:, self.bid_index] < 4).all())
        result = torch.zeros(bs, self.num_action).cuda()
        count = 0
        xi = x
        mask = x[:, self.bid_index] == 0
        for linear in self.linears:
            i = count // 5
            j = count % 5
            if j == 0:
                mask = x[:, self.bid_index] == i
                if i == 0:
                    xi = x.narrow(1, 0, self.hand_dim)
                else:
                    xi = x.narrow(1, 0, self.bid_index)
            xi = linear(xi)
            xi = self.leaky_relu(xi)
            if j == 4:
                result += xi * mask.float().unsqueeze(1).expand_as(xi)
            count += 1

        '''
        for i in range(4):
            mask = x[:, self.bid_index] == i
            if i == 0:
                xi = x.narrow(1, 0, self.hand_dim)
            else:
                xi = x.narrow(1, 0, self.bid_index)
            for j in range(5):
                xi = self.linears[i * 5 + j](xi)
                xi = nn.LeakyReLU(0.2)(xi)
            result += xi * mask.float().unsqueeze(1).expand_as(xi)
            '''
        return {"pi": result.detach().cpu()}
