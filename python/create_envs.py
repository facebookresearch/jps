import set_path

set_path.append_sys_path()

import os
import pprint

import torch
import rela
import bridge

assert rela.__file__.endswith(".so")
assert bridge.__file__.endswith(".so")


def create_train_env(
    method,
    seed,
    num_thread,
    num_game_per_thread,
    actor_cons,
    max_len
):
    assert method in ["iql"]
    context = rela.Context()
    games = []
    actors = []
    threads = []
    num_player = 4
    for thread_idx in range(num_thread):
        env = rela.VectorEnv()
        for game_idx in range(num_game_per_thread):
            unique_seed = seed + game_idx + thread_idx * num_game_per_thread
            game = bridge.BridgeEnv(
                {
                    "data_prefix": "/checkpoint/qucheng/bridge/random_data/dda",
                    "thread_idx": str(thread_idx * num_game_per_thread + game_idx),
                    "seed": str(thread_idx * num_game_per_thread + game_idx),
                    "num_thread": str(num_thread * num_game_per_thread),
                    "feature_version": "batched"
                },
                False,
            )
            games.append(game)
            env.append(game)

        assert max_len > 0

        assert len(actor_cons) == num_player
        env_actors = []
        for i in range(num_player):
            env_actors.append(actor_cons[i](thread_idx))
        actors.extend(env_actors)
        thread = bridge.BridgeThreadLoop(env_actors, env, False)

        threads.append(thread)
        context.push_env_thread(thread)
    print(
        "Finished creating environments with %d games and %d actors"
        % (len(games), len(actors))
    )
    return context, games, actors, threads


def create_eval_env(
    seed,
    num_thread,
    actor_cons,
    eval_eps,
    log_prefix=None,
):
    context = rela.Context()
    games = []
    for i in range(num_thread):
        num_player = 4
        game = bridge.BridgeEnv(
            {"data_prefix": "/checkpoint/qucheng/bridge/random_data/test",
             "thread_idx": str(i),
             "num_thread": str(num_thread),
             "seed": str(i),
             "feature_version": "batched"
            },
            False,
        )
        games.append(game)
        env = rela.VectorEnv()
        env.append(game)
        env_actors = []
        assert(len(actor_cons) == 4)
        for j in range(num_player):
            env_actors.append(actor_cons[j]())
        if log_prefix is None:
            thread = bridge.BridgeThreadLoop(env_actors, env, True)
        else:
            log_file = os.path.join(log_prefix, "game%d.txt" % i)
            thread = bridge.BridgeThreadLoop(env_actors, env, True, log_file)
        context.push_env_thread(thread)
    return context, games
