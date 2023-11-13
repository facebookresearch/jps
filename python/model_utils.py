import torch
import torch.nn as nn

import math
import hashlib


class Swish(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x):
        return x * torch.sigmoid(x)


class LinearBlock(torch.jit.ScriptModule):
    def __init__(self, dim, act="relu"):
        super().__init__()

        if act == "relu":
            act_module = nn.ReLU
        elif act == "swish":
            act_module = Swish
        elif act == "gelu":
            act_module = nn.GELU
        elif act == "tanh":
            act_module = nn.Tanh

        layers = []
        layers.append(nn.Linear(dim, dim))
        # layers.append(nn.ReLU(inplace=True))
        layers.append(act_module())
        layers.append(nn.Linear(dim, dim))
        # layers.append(nn.ReLU(inplace=True))
        layers.append(act_module())
        self.fc = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, x):
        return self.fc(x) + x


class LinearList(torch.jit.ScriptModule):
    def __init__(self, dim, num_layer, act="relu"):
        super().__init__()
        layers = []
        for _ in range(num_layer):
            layers.append(LinearBlock(dim, act))
        self.fc = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, x):
        return self.fc(x)


class LinearBlock2(torch.jit.ScriptModule):
    def __init__(self, dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.BatchNorm1d(dim))
        # layers.append(nn.LayerNorm(dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.BatchNorm1d(dim))
        # layers.append(nn.LayerNorm(dim))
        layers.append(nn.GELU())
        self.fc = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, x):
        return self.fc(x) + x


class LinearList2(torch.jit.ScriptModule):
    def __init__(self, dim, num_layer):
        super().__init__()
        layers = []
        for i in range(num_layer):
            layers.append(LinearBlock2(dim))
        self.fc = nn.Sequential(*layers)

    @torch.jit.script_method
    def forward(self, x):
        return self.fc(x)


def get_md5(filename):
    md5_hash = hashlib.md5()
    with open(filename, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
        return md5_hash.hexdigest()


# TODO: make it a decorator.
def load_model_if_exists(kwargs, initializer, attributes_to_save=[]):
    load_model = kwargs.get("load_model", None)
    if load_model is not None:
        print(f"Loading {load_model}, md5: {get_md5(load_model)}")
        data = torch.load(load_model)
        init_args = data.get("init_args", kwargs)
        # Replace load_model.
        init_args["load_model"] = load_model

        # if init_args doesn't have a key, fall back to kwargs.
        for k, v in kwargs.items():
            if k not in init_args:
                init_args[k] = v

    else:
        kwargs["use_old_feature"] = kwargs.get("use_old_feature", False)
        init_args = kwargs

    net = initializer(**init_args)
    net._attributes_to_save = attributes_to_save

    if load_model is not None:
        if "state_dict" in data:
            net.load_state_dict(data["state_dict"])

            if "attributes" in data:
                for a in attributes_to_save:
                    # Load these attributes.
                    v = data["attributes"][a]
                    print(f"Loading attr={a}: {v}")
                    setattr(net, a, v)
        else:
            # Old version, data = state_dict
            net.load_state_dict(data)


def clone_model(model, device):
    # Don't copy load_model
    args = dict(model.init_args)
    args["load_model"] = None
    args["device"] = device
    cloned = model.__class__(**args)
    cloned.load_state_dict(model.state_dict())

    for a in model._attributes_to_save:
        setattr(cloned, a, getattr(model, a))

    return cloned.to(device)


def save_model(model, filename):
    attributes = dict()
    for a in model._attributes_to_save:
        attributes[a] = getattr(model, a)

    data = {
        "state_dict": model.state_dict(),
        "init_args": model.init_args,
        "attributes": attributes
    }

    torch.save(data, filename)
