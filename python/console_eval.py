import argparse
import logging
import os
import sys
import time

import hydra
import numpy as np
import torch

import a2c
import bridge
import common_utils
import rela
import utils

from create_envs2 import create_eval_env, create_train_env
from model_utils import get_md5
from sgd_changed import SGDChanged
from trainer import Trainer


def evaluate_models(
    epoch, args, factory, actor_gen, actor_gen_baseline, env_actor_gen, stat
):
    trainer = Trainer()
    trainer.initialize(actor_gen, actor_gen_baseline)

    context = rela.Context()
    games, env_actors = create_eval_env(
        epoch, context, args, factory, trainer, env_actor_gen
    )

    console_messenger = bridge.ConsoleMessenger.get_messenger()
    if console_messenger:
        console_messenger.start()

    context.start()
    while not context.terminated():
        time.sleep(0.5)

    context.terminate()
    while not context.terminated():
        time.sleep(0.5)

    for ea in env_actors:
        for r in ea.get_history_rewards():
            for i, rr in enumerate(r):
                stat[f"eval_score_p{i}"].feed(rr)


@hydra.main(config_path="conf/eval.yaml", strict=True)
def main(args):
    # torch.backends.cudnn.benchmark = True
    logging.info(common_utils.init_context(args))
    save_dir = "./"

    logging.info("Initialize factory")
    factory = hydra.utils.instantiate(args.game)
    factory.set_args(args)

    env_actor_gen = hydra.utils.instantiate(args.env_actor_gen)
    game_info = utils.get_game_info(args, factory, env_actor_gen)

    # Setting missing parameters.
    # [TODO] Is there a better way to do it?
    args.agent.params.input_dim = game_info["input_dim"]
    args.agent.params.num_action = game_info["num_action"]
    if "agent" in args.baseline:
        args.baseline.agent.params.input_dim = game_info["input_dim"]
        args.baseline.agent.params.num_action = game_info["num_action"]

    # Construct agent.
    actor_gen = hydra.utils.instantiate(args.actor_gen)
    agent = hydra.utils.instantiate(args.agent)

    actor_gen_baseline = hydra.utils.instantiate(args.baseline.actor_gen)
    if "agent" in args.baseline:
        agent_baseline = hydra.utils.instantiate(args.baseline.agent)

    actor_gen.initialize(agent)
    if "agent" in args.baseline:
        actor_gen_baseline.initialize(agent_baseline)

    stat = common_utils.MultiCounter("./")
    stat.reset()
    evaluate_models(
        args.eval_fake_epoch,
        args,
        factory,
        actor_gen,
        actor_gen_baseline,
        env_actor_gen,
        stat,
    )
    logging.info(stat.summary(args.eval_fake_epoch))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
