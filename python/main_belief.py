import time
import os
import sys
import argparse

import numpy as np
import torch

from create_envs2 import create_train_env
import a2c
import common_utils
import rela
import utils
import hydra

from sgd_changed import SGDChanged

import logging
log = logging.getLogger(__file__)


def make_optimizer(params, optim, lr, momentum=None, eps=None):
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)

    if optim == "sgd_changed":
        return SGDChanged(params, lr=lr, momentum=momentum)

    if optim == "adam":
        return torch.optim.Adam(params, lr=lr, eps=eps)

    return None


@hydra.main(config_path="conf/config_belief.yaml", strict=True)
def main(args):
    # torch.backends.cudnn.benchmark = True
    log.info(common_utils.init_context(args))
    save_dir = "./"

    factory = hydra.utils.instantiate(args.game)
    factory.set_args(args)

    env_actor_gen = hydra.utils.instantiate(args.env_actor_gen)
    game_info = utils.get_game_info(args, factory, env_actor_gen)

    # Setting missing parameters.
    # [TODO] Is there a better way to do it?
    args.agent.params.input_dim = game_info["input_dim"] - 1
    args.agent.params.num_action = game_info["num_action"]

    # Construct agent.
    actor_gen = hydra.utils.instantiate(args.actor_gen)
    agent = hydra.utils.instantiate(args.agent)
    agent = agent.to(args.train_device)
    log.info(agent)

    print("Create belief model")
    args.belief_model.params.input_dim = game_info["input_dim"] - 1
    args.belief_model.params.num_action = game_info["num_action"]

    belief_model = hydra.utils.instantiate(args.belief_model)
    belief_model = belief_model.to(args.train_device)
    belief_model.reset_lstm()
    belief_trainable_params = lambda: belief_model.parameters()

    print("Create weight model")
    args.weight_model.params.input_dim = game_info["input_dim"] - 1

    weight_model = hydra.utils.instantiate(args.weight_model)
    weight_model = weight_model.to(args.train_device)
    weight_trainable_params = lambda: weight_model.parameters()

    # Let's train the model otherwise.
    # if args.negative_momentum is not None:
    #     # Negative momentum!
    #     optim = SGDChanged(trainable_params(),
    #                        lr=args.lr,
    #                        momentum=args.negative_momentum)
    # else:
    #     optim = torch.optim.Adam(trainable_params(), lr=args.lr, eps=args.eps)

    belief_optim = make_optimizer(belief_trainable_params(),
                                  optim=args.belief_optim,
                                  lr=args.belief_lr,
                                  momentum=args.momentum,
                                  eps=args.eps)

    weight_optim = make_optimizer(weight_trainable_params(),
                                  optim=args.weight_optim,
                                  lr=args.weight_lr,
                                  momentum=args.momentum,
                                  eps=args.eps)

    # belief_scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     belief_optim,
    #     base_lr=args.min_lr,
    #     max_lr=args.belief_lr,
    #     step_size_up=args.epoch_len // 2)
    # weight_scheduler = torch.optim.lr_scheduler.CyclicLR(
    #     weight_optim,
    #     base_lr=args.min_lr,
    #     max_lr=args.weight_lr,
    #     step_size_up=args.epoch_len // 2)
    belief_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        belief_optim, 'min', factor=0.5, min_lr=args.min_lr, verbose=True)
    weight_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        weight_optim, 'min', factor=0.5, min_lr=args.min_lr, verbose=True)

    # Create replay_buffer.
    print("Create replay buffer")
    replay_buffer = rela.PrioritizedReplay2(args.replay_buffer_size, args.seed,
                                            args.priority_exponent,
                                            args.priority_weight,
                                            args.prefetch, 0)

    actor_gen.set_replay_buffer(replay_buffer)
    actor_gen.initialize(agent)

    print("Create trainer")
    # Create models for 2 players.
    trainer = hydra.utils.instantiate(args.trainer)
    trainer.initialize(actor_gen, actor_gen)

    # Then we create the environment.
    context = rela.Context()

    print("Create train env")
    create_train_env(
        context,
        args,
        factory,
        trainer,
        env_actor_gen,
        replay_buffer
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

    belief_stat = common_utils.MultiCounter(save_dir)
    weight_stat = common_utils.MultiCounter(save_dir)

    tachometer = utils.Tachometer()
    train_time = 0
    for epoch in range(args.num_epoch):
        mem_usage = common_utils.get_mem_usage()
        log.info("Beginning of Epoch %d\nMem usage: %s" % (epoch, mem_usage))

        belief_stat.reset()
        weight_stat.reset()

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

            belief_loss, belief_stat_dict = belief_model.loss(batch.d)
            weight_loss = weight_model.loss(batch.d)

            # log.info(loss.size())
            # loss = (loss * weight).mean()

            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("calculating loss")

            if not args.eval_only:
                belief_optim.zero_grad()
                belief_loss.backward()
                belief_g_norm = torch.nn.utils.clip_grad_norm_(
                    belief_trainable_params(), args.grad_clip)
                belief_optim.step()
                belief_stat["grad_norm"].feed(belief_g_norm)

                weight_optim.zero_grad()
                weight_loss.backward()
                weight_g_norm = torch.nn.utils.clip_grad_norm_(
                    weight_trainable_params(), args.grad_clip)
                weight_optim.step()
                weight_stat["grad_norm"].feed(weight_g_norm)

                # belief_scheduler.step()
                # weight_scheduler.step()

            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("backprop & update")

            replay_buffer.keep_priority()

            # replay_buffer.update_priority(priority)
            # if stopwatch is not None:
            #    stopwatch.time("updating priority")

            belief_stat["loss"].feed(belief_loss.detach().item())
            for k, v in belief_stat_dict.items():
                belief_stat[k].feed(v)

            weight_stat["loss"].feed(weight_loss.detach().item())

        belief_scheduler.step(belief_stat["loss"].mean())
        weight_scheduler.step(weight_stat["loss"].mean())

        epoch_t = time.time() - t
        train_time += epoch_t
        log.info("\nepoch: %d, time: %.1fs, total time(train): %s" %
                 (epoch, epoch_t, common_utils.sec2str(train_time)))
        # [TODO] Fix tachometer
        log.info('\n' + tachometer.lap(None, replay_buffer, args.epoch_len *
                                       args.batchsize, 1))
        if stopwatch is not None:
            log.info('\n' + stopwatch.summary())

        belief_model.save(f"belief-{epoch}.pth")
        weight_model.save(f"weight-{epoch}.pth")

        # log.info(stat.summary(epoch))
        # log.info("****************************************")
        log.info("belief_stat")
        log.info(belief_stat.summary(epoch))
        log.info("weight_stat")
        log.info(weight_stat.summary(epoch))
        log.info("****************************************")


if __name__ == "__main__":
    main()
