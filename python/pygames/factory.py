import sys
sys.path.append("..")

import os
import time

import set_path
import random
set_path.append_sys_path()

from create_envs2 import create_eval_env

import rela
assert rela.__file__.endswith(".so")

import simple_game
assert simple_game.__file__.endswith(".so")

class Factory:
    def __init__(self, game_type, step, N):
        self.game_type = game_type
        self.step = step
        self.N = N

    def set_args(self, args):
        self.args = args

    def create_game(self, options):
        if self.game_type == "comm":
            return simple_game.Communicate(options) 
        elif self.game_type == "comm2":
            return simple_game.Communicate2(options) 
        elif self.game_type == "simplebidding":
            return simple_game.SimpleBidding(options) 
        elif self.game_type == "2suitedbridge":
            return simple_game.TwoSuitedBridge(options) 
        else:
            raise NotImplementedError(f"game_type: {self.game_type} not implemented")

    def create_train_env(self, thread_idx, game_idx):
        args = self.args
        unique_seed = args.seed + game_idx + thread_idx * args.num_game_per_thread

        options = simple_game.CommOptions()
        options.num_round = self.step
        options.N = self.N
        return self.create_game(options)

    def create_eval_env(self, epoch, thread_idx):
        args = self.args
        unique_seed = args.seed * 2341 + 5234 + thread_idx + epoch * 97142

        options = simple_game.CommOptions()
        options.num_round = self.step
        options.N = self.N
        options.seq_enumerate = True
        return self.create_game(options)

