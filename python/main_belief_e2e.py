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

sys.path.append("../dds")
import card_utils
from sample_utils import DealWalk, get_future_tricks

from belief import *

def make_optimizer(params, optim, lr, momentum=None, eps=None):
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)

    if optim == "sgd_changed":
        return SGDChanged(params, lr=lr, momentum=momentum)

    if optim == "adam":
        return torch.optim.Adam(params, lr=lr, eps=eps)

    return None


def get_samples(belief_model, obs, sample_size):
    init_rep = belief_model.get_init_rep(obs).detach()
    batch_size = init_rep.size(0)
    init_rep = init_rep.unsqueeze(1).expand(-1, sample_size, -1).contiguous()
    cards = obs["cards"].detach()
    cards = cards.unsqueeze(1).expand(-1, sample_size, -1).contiguous()
    ret = belief_model.sample(init_rep.view(batch_size * sample_size, -1),
                              cards.view(batch_size * sample_size, -1))
    samples = ret["cards"].view(batch_size, sample_size, -1)
    return samples


def get_tricks(cards, strain, first=0):
    if cards.dim() == 3:
        batch_size, sample_size, _ = cards.size()
        cards_list = cards.detach().view(batch_size * sample_size, -1).tolist()
        strain = strain.detach().unsqueeze(-1).expand(
            -1, sample_size).contiguous()
        strain_list = strain.view(batch_size * sample_size).tolist()
    else:
        cards_list = cards.detach().tolist()
        strain_list = strain.detach().tolist()

    deals = [DealWalk.from_card_map(c) for c in cards_list]
    tricks, _ = get_future_tricks(deals, strain_list, first)

    tricks = torch.from_numpy(tricks).float().reshape_as(cards).to(
        cards.device)
    return tricks


@hydra.main(config_path="conf/config_belief_e2e.yaml", strict=True)
def main(args):
    # torch.backends.cudnn.benchmark = True
    log.info(common_utils.init_context(args))
    save_dir = "./"

    factory = hydra.utils.instantiate(args.game)
    factory.set_args(args)

    game_info = utils.get_game_info(factory)

    # Setting missing parameters.
    # [TODO] Is there a better way to do it?
    args.agent.params.input_dim = game_info["input_dim"]
    args.agent.params.num_action = game_info["num_action"]

    print(f"input_dim = {args.agent.params.input_dim}")

    # Construct agent.
    actor_gen = hydra.utils.instantiate(args.actor_gen)
    agent = hydra.utils.instantiate(args.agent)
    agent = agent.to(args.train_device)
    log.info(agent)

    print("Create belief model")
    args.belief_model.params.input_dim = game_info["input_dim"]
    args.belief_model.params.num_action = game_info["num_action"]

    assert args.belief_model.params.input_dim == 1653

    # belief_model = hydra.utils.instantiate(args.belief_model)
    # belief_model = belief_model.to(args.train_device)
    # belief_model.reset_lstm()
    # belief_trainable_params = lambda: belief_model.parameters()
    belief_model_class = LSTMModel
    belief_model_params = torch.load(args.belief_model_path)
    belief_model = belief_model_class(load_model=args.belief_model_path)
    belief_model.to(args.train_device)
    belief_model.eval()
    belief_model.share_memory()


    print("Create weight model")
    args.weight_model.params.input_dim = game_info["input_dim"]

    assert args.weight_model.params.input_dim == 1653

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

    # belief_optim = make_optimizer(belief_trainable_params(),
    #                               optim=args.belief_optim,
    #                               lr=args.belief_lr,
    #                               momentum=args.momentum,
    #                               eps=args.eps)

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
    # belief_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     belief_optim, 'min', factor=0.5, min_lr=args.min_lr, verbose=True)
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
    create_train_env(context, args, factory, trainer, replay_buffer)

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

    # belief_stat = common_utils.MultiCounter(save_dir)
    weight_stat = common_utils.MultiCounter(save_dir)

    tachometer = utils.Tachometer()
    train_time = 0
    for epoch in range(args.num_epoch):
        mem_usage = common_utils.get_mem_usage()
        log.info("Beginning of Epoch %d\nMem usage: %s" % (epoch, mem_usage))

        # belief_stat.reset()
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

            # belief_loss, belief_stat_dict = belief_model.loss(batch.d)

            strain = batch.d["strain"].detach().long()

            print(f"strain = {strain}")

            samples = get_samples(belief_model, batch.d, args.sample_size)
            tricks_gt = get_tricks(batch.d["cards"], strain, 0)
            tricks = get_tricks(samples, strain, 0)

            obs = dict(cards=samples,
                       s=batch.d["s"],
                       tricks_gt=tricks_gt,
                       tricks=tricks)
            weight_loss = weight_model.loss(obs)

            # log.info(loss.size())
            # loss = (loss * weight).mean()

            if stopwatch is not None:
                torch.cuda.synchronize()
                stopwatch.time("calculating loss")

            if not args.eval_only:
                # belief_optim.zero_grad()
                # belief_loss.backward()
                # belief_g_norm = torch.nn.utils.clip_grad_norm_(
                #     belief_trainable_params(), args.grad_clip)
                # belief_optim.step()
                # belief_stat["grad_norm"].feed(belief_g_norm)

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

            # belief_stat["loss"].feed(belief_loss.detach().item())
            # for k, v in belief_stat_dict.items():
            #     belief_stat[k].feed(v)

            weight_stat["loss"].feed(weight_loss.detach().item())

        # belief_scheduler.step(belief_stat["loss"].mean())
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

        # belief_model.save(f"belief-{epoch}.pth")
        weight_model.save(f"weight-{epoch}.pth")

        # log.info(stat.summary(epoch))
        # log.info("****************************************")
        # log.info("belief_stat")
        # log.info(belief_stat.summary(epoch))
        log.info("weight_stat")
        log.info(weight_stat.summary(epoch))
        log.info("****************************************")


if __name__ == "__main__":
    main()
