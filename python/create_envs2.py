import set_path
import random
from collections import deque

from env_actor_gen import EnvActorGen

set_path.append_sys_path()

import os
import pprint

import torch
import rela
from trainer import BaseTrainer

import time
assert rela.__file__.endswith(".so")

def create_train_env(
    context,
    args, 
    factory,
    trainer : BaseTrainer,
    env_actor_gen: EnvActorGen,
    replay_buffer
):
    threads = []

    for thread_idx in range(args.num_thread):
        env_actors = []

        for game_idx in range(args.num_game_per_thread):
            game = factory.create_train_env(thread_idx, game_idx)
            seed = thread_idx * 123 + game_idx + args.seed

            # Create env_actors given the models. 
            actors = trainer.gen_actors("train", thread_idx, game.spec(), args.seed)
            ea = env_actor_gen.generate(thread_idx, seed, game, actors, args, False, replay_buffer)
            env_actors.append(ea)

        # -1: infinite game loop.
        thread = rela.ThreadLoopEnvActor(thread_idx, env_actors, -1)
        threads.append(thread)
        context.push_env_thread(thread)

    print(
        "Finished creating environments with %d games and %d threads"
        % (args.num_thread * args.num_game_per_thread, args.num_thread)
    )

def create_eval_env(
    epoch,
    context,
    args, 
    factory,
    trainer : BaseTrainer,
    env_actor_gen: EnvActorGen
):
    threads = []
    games = []
    env_actors = []

    eval_first_k = args.eval_first_k if "eval_first_k" in args else -1 

    for thread_idx in range(args.num_thread):
        game = factory.create_eval_env(epoch, thread_idx)
        games.append(game)

        # Create env_actors given the models. 
        seed = args.seed + epoch * 1523
        actors = trainer.gen_actors("eval", thread_idx, game.spec(), seed)
        ea = env_actor_gen.generate(thread_idx, seed, game, actors, args, True, None)
        env_actors.append(ea)

        # Eval sequentially until the bot uses all the data. 
        thread = rela.ThreadLoopEnvActor(thread_idx, env_actors[-1:], eval_first_k)
        threads.append(thread)
        context.push_env_thread(thread)

    print(f"Finished creating {args.num_thread} eval environments")

    return games, env_actors

