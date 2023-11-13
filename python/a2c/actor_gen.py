import sys
sys.path.append("..")

import utils

import rela
assert rela.__file__.endswith(".so")

# Actor constructors.
class A2CActorGen:
    def __init__(self, **kwargs):
        utils.set_default(kwargs, "multi_step", 1)
        utils.set_default(kwargs, "gamma", 0.99)
        utils.set_default(kwargs, "use_sampling_in_eval", False)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.init_args = kwargs

        self.initialized = False
        self.rb = None
        self.model_servers_train = []
        self.model_servers_eval = None
        self.model_lockers = []
        self.devices = self.device.split(',')

    def set_replay_buffer(self, rb):
        self.rb = rb

    def initialize(self, agent):
        for device in self.devices:
            self._initialize(agent, device)
        self.initialized = True

    def _initialize(self, agent, device):
        ref_model = [agent.clone(device) for _ in range(3)]
        model_locker = rela.ModelLocker(ref_model, device)
        self.model_lockers.append(model_locker)

        self._set_model_servers(device)

    def _set_model_servers(self, device):
        if self.model_servers_eval is None:
            model_servers_eval = rela.Models()
            func_bind = "act" if self.use_sampling_in_eval else "act_greedy"
            # print(f"In eval model with {func_bind}")
            model_servers_eval.add(
                "act",
                rela.BatchProcessor(self.model_lockers[0], func_bind, self.batchsize, device)
            )
            self.model_servers_eval = model_servers_eval

        if self.rb is None:
            return

        # Also set train
        model_servers = rela.Models()
        model_servers.add(
            "act",
            rela.BatchProcessor(self.model_lockers[-1], "act", self.batchsize, device)
        )
        model_servers.add(
            "compute_priority",
            rela.BatchProcessor(
                self.model_lockers[-1],
                "compute_priority",
                self.batchsize,
                device)
        )

        # Set train servers.
        if self.model_servers_train is None:
            self.model_servers_train = [model_servers]
        else:
            self.model_servers_train.append(model_servers)

    def gen_actor(self, train_or_eval, thread_idx):
        assert self.initialized, "actor_gen is not initialized!"

        if train_or_eval == "train":
            m = self.model_servers_train[thread_idx % len(self.model_servers_train)]
            rb = self.rb
        elif train_or_eval == "eval":
            assert self.model_servers_eval is not None, "self.model_servers_eval should be valid!"
            m = self.model_servers_eval
            rb = None
        else:
            raise RuntimeError(f"Unknown train_or_eval in gen_actor! {train_or_eval}")

        return rela.A2CActor(m, self.multi_step, self.gamma, rb)

    def update(self, agent):
        assert self.initialized, "actor_gen is not initialized!"

        for m in self.model_lockers:
            m.update_model(agent)
