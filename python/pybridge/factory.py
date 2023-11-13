from .baseline_model import BaselineBridgeModel

import sys

sys.path.append("..")

import os
import time
import torch

import set_path
import random

set_path.append_sys_path()

from a2c.actor_gen import A2CActorGen
from create_envs2 import create_eval_env

import rela

assert rela.__file__.endswith(".so")

import bridge

assert bridge.__file__.endswith(".so")


class BaselineActorGen:
    def __init__(self, batchsize, device):
        models = rela.Models()
        ref_model = [BaselineBridgeModel()]
        model_locker = rela.ModelLocker(ref_model, device)

        models.add("act", rela.BatchProcessor(model_locker, "act", batchsize, device))

        self.models = models
        self.model_locker = model_locker

    def gen_actor(self, train_or_eval, thread_idx):
        return bridge.BaselineActor2(self.models)


class AllPassActorGen:
    def __init__(self):
        pass

    def gen_actor(self, train_or_eval, thread_idx):
        return bridge.AllPassActor2()


class RandomActorGen:
    def __init__(self):
        pass

    def gen_actor(self, train_or_eval, thread_idx):
        return bridge.RandomActor()


class GreedyPlayActorGen:
    def __init__(self, device=None):
        self.device = device
        self.model_locker = None
        self.model_server = None

    def initialize(self, agent=None):
        if agent is not None:
            assert self.device is not None
            ref_models = [agent.clone(self.device)]
            self.model_locker = rela.ModelLocker(ref_models, self.device)
            self.model_server = rela.Models()
            func_bind = "act_greedy"
            self.model_server.add("act", rela.BatchProcessor(
                self.model_locker, func_bind, 1, self.device))

    def gen_actor(self, train_or_eval, thread_idx):
        if self.model_server is None:
            return bridge.GreedyPlayActor()
        return bridge.GreedyPlayActor(self.model_server)


class ConsoleActorGen:
    def __init__(self):
        pass

    def gen_actor(self, train_or_eval, thread_idx):
        return bridge.ConsoleActor(thread_idx)


def create_single_env(db_filename, thread_idx, seed, num_thread=50):
    db = bridge.DBInterface(db_filename, "", num_thread)

    params = {
        "thread_idx": str(thread_idx),
        "seed": str(seed),
        "feature_version": "single",
    }
    env = bridge.BridgeEnv(db, params, False)

    return db, env


class Factory:
    # feature_version can be single, old, batch
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.init_args = {k: str(v) for k, v in kwargs.items()}

        # Initialize database.
        num_train_games = self.num_thread * self.num_game_per_thread
        num_test_games = self.num_thread

        self.train_db = bridge.DBInterface(
            self.train_dataset, self.train_save_db, num_train_games
        )
        self.test_db = bridge.DBInterface(
            self.test_dataset, self.test_save_db, num_test_games
        )

        print(
            f"Train db size: {self.train_db.get_dataset_size()}, #train_games: {num_train_games}"
        )
        print(
            f"Test db size: {self.test_db.get_dataset_size()}, #test_games: {num_test_games}"
        )

    def set_args(self, args):
        # [TODO] Hack for shared parameters.
        self.args = args

        # init console messenger
        if "console_messenger_type" in args:
            verbose = (
                args.console_messenger_verbose
                if args.console_messenger_verbose
                else False
            )
            if args.console_messenger_type == "socket":
                port = (
                    args.console_messenger_port if args.console_messenger_port else 2001
                )
                bridge.ConsoleMessenger.init_messenger(
                    {"type": "socket", "port": str(port), "verbose": str(verbose)}
                )
            else:
                bridge.ConsoleMessenger.init_messenger(
                    {"type": "cmdline", "verbose": str(verbose)}
                )
            bridge.ConsoleMessenger.get_messenger().start()

    def create_train_env(self, thread_idx, game_idx):
        args = self.args
        unique_seed = args.seed + game_idx + thread_idx * args.num_game_per_thread

        params = {
            "thread_idx": str(thread_idx * args.num_game_per_thread + game_idx),
            "seed": str(unique_seed),
            "eval_mode": "false"
        }

        params.update(self.init_args)
        
        if self.train_playing:
            if self.train_bidding:
                return bridge.DuplicateBridgeEnv(params)
            else:
                return bridge.DuplicateBridgeEnv(self.train_db, params)

        return bridge.BridgeEnv(self.train_db, params, False)

    def create_eval_env(self, epoch, thread_idx):
        args = self.args
        unique_seed = args.seed * 2341 + 5234 + thread_idx + epoch * 97142

        params = {
            "thread_idx": str(thread_idx),
            "seed": str(unique_seed),
            "eval_mode": "true"
        }

        if "console_eval" in args and args.console_eval:
            params["console_eval"] = "true"

        params.update(self.init_args)

        # Override. Eval env uses seq sampler by default.
        params["sampler"] = "seq"

        if self.train_playing:
            # if self.train_bidding:
            #     return bridge.DuplicateBridgeEnv(params)
            return bridge.DuplicateBridgeEnv(self.test_db, params)

        return bridge.BridgeEnv(self.test_db, params, False)
