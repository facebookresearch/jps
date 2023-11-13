import torch
import collections

model = torch.load("/checkpoint/qucheng/bridge/1107/2_blocks_2_hidden_200_0.001/model_198000.bin")

for key in ("linear_policy", "linear_goal"):
    model[key + ".weight"] = torch.cat([ model[key + ".weight"], torch.FloatTensor(1, 200).fill_(0.0).cuda() ], dim=0)
    model[key + ".bias"] = torch.cat([ model[key + ".bias"], torch.FloatTensor(1).fill_(-10).cuda() ], dim=0)

model = collections.OrderedDict([ ("online_net." + k, v) for k, v in model.items() ]) 
init_args = dict(hid_dim=200, num_block=2, use_old_feature=True)

save_filename = "bridge_old_model_converted.pth" 
torch.save(dict(state_dict=model, init_args=init_args), save_filename)
print(f"Save to {save_filename}")
