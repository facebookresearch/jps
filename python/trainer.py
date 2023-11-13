from abc import ABC, abstractmethod

import set_path
import random
from collections import deque

set_path.append_sys_path()

import os
import pprint

import time

import torch
import rela
assert rela.__file__.endswith(".so")

class ModelPool:
    def __init__(self, hist_len):
        self.models = deque([], hist_len)
    
    def add(self, model):
        self.models.append(model)

    def sample(self):
        # Random sample a model. 
        idx = random.randint(0, len(self.models) - 1)
        return self.models[idx]


# Different trainer that initializes actors differently. 
class BaseTrainer(ABC):
    def __init__(self):
        self.num_update = 0

    @abstractmethod
    def on_update(self, agent):
        self.num_update += 1

    @abstractmethod
    def gen_actors(self, train_or_eval, thread_idx, spec, seed):
        pass

    def on_epoch_finish(self, agent):
        pass


class Trainer(BaseTrainer):
    def __init__(self, actor_sync_freq = 0):
        super().__init__()
        self.actor_sync_freq = actor_sync_freq

    def initialize(self, actor_gen, actor_gen_opp=None):
        self.actor_gen = actor_gen
        self.actor_gen_opp = actor_gen_opp
        if self.actor_gen_opp is None:
            self.actor_gen_opp = actor_gen

    def gen_actors(self, train_or_eval, thread_idx, spec, seed):
        actors = []
        for i, p in enumerate(spec.players):
            if p == rela.PlayerGroup.GRP_NATURE:
                actors.append(rela.RandomActor(spec.max_num_actions[i], seed + 2377 * i))
            elif p == rela.PlayerGroup.GRP_1:
                actors.append(self.actor_gen.gen_actor(train_or_eval, thread_idx))
            elif p == rela.PlayerGroup.GRP_2:
                actors.append(self.actor_gen_opp.gen_actor(train_or_eval, thread_idx))
            else:
                raise RuntimeError("Unsupported player type! " + str(p))

        return actors

    def on_update(self, agent):
        if self.actor_sync_freq > 0 and self.num_update % self.actor_sync_freq == 0:
            self.actor_gen.update(agent)

        super().on_update(agent)

    def on_epoch_finish(self, agent):
        # Need to update to make sure the evaluated model is always the latest.
        self.actor_gen.update(agent)


class PBTTrainer(BaseTrainer):
    def __init__(self, model_pool_size, actor_sync_freq, actor_opponent_sync_freq):
        super().__init__()
        self.model_pool_size = model_pool_size
        self.actor_sync_freq = actor_sync_freq
        self.actor_opponent_sync_freq = actor_opponent_sync_freq

    def initialize(self, actor_gen, actor_gen_opp, act_device):
        self.act_device = act_device
        self.model_pool = ModelPool(self.model_pool_size)
        # self.model_pool.add(agent.clone(act_device))

        # Self model
        self.actor_gen = actor_gen
        # Opponent model, no replay buffer is needed.
        self.actor_gen_opp = actor_gen_opp

    def gen_actors(self, train_or_eval, thread_idx, spec, seed):
        # NS: self model, EW: opponent model
        actors = []
        for i, p in enumerate(spec.players):
            if p == rela.PlayerGroup.GRP_NATURE:
                actors.append(rela.RandomActor(spec.max_num_actions[i], seed + 2377 * i))
            elif p == rela.PlayerGroup.GRP_1:
                actors.append(self.actor_gen.gen_actor(train_or_eval, thread_idx))
            elif p == rela.PlayerGroup.GRP_2:
                actors.append(self.actor_gen_opp.gen_actor(train_or_eval, thread_idx))
            else:
                raise RuntimeError("Unsupported player type! " + str(p))

        return actors
        
    def on_update(self, agent):
        if len(self.model_pool.models) == 0:
            self.model_pool.add(agent.clone(self.act_device))

        if self.num_update % self.actor_sync_freq == 0:
            self.actor_gen.update(agent)

        if self.num_update % self.actor_opponent_sync_freq == 0:
            opponent = self.model_pool.sample()
            self.actor_gen_opp.update(opponent)

        super().on_update(agent)

    def on_epoch_finish(self, agent):
        self.actor_gen.update(agent)
        self.model_pool.add(agent.clone(self.act_device))


