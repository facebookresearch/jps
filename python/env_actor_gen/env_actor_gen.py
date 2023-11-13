# Generate different env actor
import set_path
import random
set_path.append_sys_path()

import os
import pprint

import torch
import rela
from trainer import BaseTrainer

import time
assert rela.__file__.endswith(".so")

import bridge
assert bridge.__file__.endswith(".so")

class EnvActorGen:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self.gen_type in ("search", "search_new"):
            search_options = rela.SearchActorOptions()
            search_options.search_ratio = self.search_ratio 
            search_options.update_count = self.update_count
            search_options.verbose_freq = self.search_verbose_freq
            search_options.baseline_ratio = self.search_baseline_ratio
            search_options.use_hacky = self.search_use_hacky
            search_options.use_tabular_ref = self.search_use_tabular_ref
            search_options.use_grad_update = self.search_use_grad_update
            search_options.best_on_best = self.search_best_on_best

            self.search_options = search_options

    def generate(self, thread_idx, seed, game, actors, args, is_eval, replay_buffer=None, empty_init=False):
        options = rela.EnvActorOptions()
        options.save_prefix = args.save_prefix
        options.display_freq = args.display_freq
        options.eval = is_eval
        options.thread_idx = thread_idx
        options.seed = seed
        options.empty_init = empty_init

        if self.gen_type == "search":
            ea = rela.SearchActor(game, actors, options, self.search_options)
        elif self.gen_type == "search_new":
            ea = rela.SearchActorNew(game, actors, options, self.search_options)
        elif self.gen_type == "belief":
            bt_options = bridge.BeliefTransferOptions()
            bt_options.debug = args.debug
            bt_options.opening_lead = args.train_opening_lead
            ea = bridge.BeliefTransferEnvActor(game, actors, options, bt_options, replay_buffer)
        elif self.gen_type == "cross_bidding":
            ea = bridge.CrossBiddingEnvActor(game, actors, options, replay_buffer)
        elif self.gen_type == "basic":
            more_options = rela.EnvActorMoreOptions()
            more_options.use_sad = self.use_sad
            ea = rela.EnvActor(game, actors, options, more_options)
        else:
            raise RuntimeError(f"Unknown gen_type: {self.gen_type}")

        return ea

