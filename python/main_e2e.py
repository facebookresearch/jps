import time
import os
import sys
import argparse

import numpy as np
import torch

from create_envs2 import create_train_env, create_eval_env
from trainer import Trainer, PBTTrainer
import a2c
import common_utils
import rela
import utils
import hydra

from model_utils import get_md5

from sgd_changed import SGDChanged

import logging
log = logging.getLogger(__file__)
#sys.stdout = utils.LoggerWriter(log.info)
#sys.stderr = utils.LoggerWriter(log.error)


def evaluate_models(epoch, args, factory, actor_gen, actor_gen_baseline,
                    env_actor_gen, stat):
    trainer = Trainer()
    trainer.initialize(actor_gen, actor_gen_baseline)

    context = rela.Context()
    games, env_actors = create_eval_env(epoch, context, args, factory, trainer,
                                        env_actor_gen)

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


@hydra.main(config_path="conf/config_e2e.yaml")
def main(args):
    # torch.backends.cudnn.benchmark = True
    log.info(common_utils.init_context(args))
    save_dir = "./"

    log.info("Initialize factory")
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
    else:
        agent_baseline = None

    if args.eval_only:
        actor_gen.initialize(agent)
        if agent_baseline is not None:
            actor_gen_baseline.initialize(agent_baseline)

        stat = common_utils.MultiCounter("./")
        stat.reset()
        evaluate_models(args.eval_fake_epoch, args, factory, actor_gen,
                        actor_gen_baseline, env_actor_gen, stat)
        log.info(stat.summary(args.eval_fake_epoch))

        sys.exit(0)

    agent = agent.to(args.train_device)
    log.info(agent)

    trainable_params = lambda: agent.online_net.parameters()

    # Let's train the model otherwise.
    if args.negative_momentum is not None:
        # Negative momentum!
        optim = SGDChanged(trainable_params(),
                           lr=args.lr,
                           momentum=args.negative_momentum)
    else:
        optim = torch.optim.Adam(trainable_params(), lr=args.lr, eps=args.eps)

    # Create replay_buffer.
    print("Create replay buffer")
    replay_buffer = rela.PrioritizedReplay2(args.replay_buffer_size, args.seed,
                                            args.priority_exponent,
                                            args.priority_weight,
                                            args.prefetch, 0)

    actor_gen.set_replay_buffer(replay_buffer)
    actor_gen.initialize(agent)

    if agent_baseline is not None:
        actor_gen_baseline.initialize(agent_baseline)

    print("Create trainer")
    # Create models for 2 players.
    trainer = hydra.utils.instantiate(args.trainer)
    trainer.initialize(actor_gen, actor_gen, args.train_device)

    # Then we create the environment.
    context = rela.Context()

    print("Create train env")
    create_train_env(
        context,
        args,
        factory,
        trainer,
        env_actor_gen,
        replay_buffer,
    )

    # Start
    print("Start context")
    context.start()
    actual_burn_in = min(args.burn_in_frames, args.replay_buffer_size)
    while replay_buffer.size() < actual_burn_in:
        log.info(
            f"warming up replay buffer: {replay_buffer.size()} / {actual_burn_in}"
        )
        time.sleep(1)

    args.record_time = True
    if args.record_time:
        stopwatch = common_utils.Stopwatch()
    else:
        stopwatch = None

    stat = common_utils.MultiCounter(save_dir)
    tachometer = utils.Tachometer()
    train_time = 0
    for epoch in range(args.num_epoch):
        mem_usage = common_utils.get_mem_usage()
        log.info("Beginning of Epoch %d\nMem usage: %s" % (epoch, mem_usage))

        stat.reset()
        if stopwatch is not None:
            stopwatch.reset()
        tachometer.start(replay_buffer.num_add())
        t = time.time()
        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if stopwatch is not None:
                torch.cuda.synchronize()

            # log.info(f"Epoch: {epoch}, Batch idx: {batch_idx}")

            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()

            trainer.on_update(agent)

            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("sync and updating")

            # no weight for a2c.
            batch, _ = replay_buffer.sample(args.batchsize, args.train_device)

            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("sample data")

            optim.zero_grad()

            # TODO a hack.
            if args.env_actor_gen.params.gen_type == "search_new":
                loss, stat_dict = agent.loss_search(batch.d)
            else:
                loss, stat_dict = agent.loss(batch.d)

            # log.info(loss.size())
            # loss = (loss * weight).mean()

            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("calculating loss")

            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(trainable_params(),
                                                    args.grad_clip)
            optim.step()

            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("backprop & update")

            replay_buffer.keep_priority()

            # replay_buffer.update_priority(priority)
            # if stopwatch is not None:
            #    stopwatch.time("updating priority")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            if "reward" in batch.d:
                stat["reward"].feed(batch.d["reward"].mean().item())
            for k, v in stat_dict.items():
                stat[k].feed(v)

        epoch_t = time.time() - t
        train_time += epoch_t
        log.info("\nepoch: %d, time: %.1fs, total time(train): %s" %
                 (epoch, epoch_t, common_utils.sec2str(train_time)))
        # [TODO] Fix tachometer
        log.info('\n' + tachometer.lap(None, replay_buffer, args.epoch_len *
                                       args.batchsize, 1))
        if stopwatch is not None:
            log.info('\n' + stopwatch.summary())

        save_filename = f"agent-{epoch}.pth"
        agent.save(save_filename)
        log.info(
            f"Save to {os.path.join(os.getcwd(), save_filename)}, md5: {get_md5(save_filename)}"
        )

        trainer.on_epoch_finish(agent)

        context.pause()
        evaluate_models(epoch, args, factory, actor_gen, actor_gen_baseline,
                        env_actor_gen, stat)
        context.resume()

        log.info(stat.summary(epoch))
        log.info("****************************************")


if __name__ == "__main__":
    main()
