import common_utils
import os
import random
import utils
import hydra

import tqdm

import glob

import torch
import torch.nn as nn

from belief import *

import logging
log = logging.getLogger(__file__)

def make_optimizer(params, optim, lr, momentum=None, eps=None):
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum)

    if optim == "adam":
        return torch.optim.Adam(params, lr=lr, eps=eps)

    return None


def batch2dict(batch):
    return { name : d.cuda() for name, d in zip(BeliefDatasetWrapper.kDataOrder, batch) }

@hydra.main(config_path="conf/offline_weight.yaml", strict=True)
def main(args):
    # torch.backends.cudnn.benchmark = True
    log.info(common_utils.init_context(args))
    factory = hydra.utils.instantiate(args.game)
    factory.set_args(args)

    game_info = utils.get_game_info_simple(factory)

    print("Create weight model")
    args.weight_model.params.input_dim = game_info["input_dim"]

    assert args.weight_model.params.input_dim == 1653

    weight_model = hydra.utils.instantiate(args.weight_model)
    weight_model = weight_model.to(args.train_device)
    weight_trainable_params = lambda: weight_model.parameters()

    weight_optim = make_optimizer(weight_trainable_params(),
                                  optim=args.weight_optim,
                                  lr=args.weight_lr,
                                  momentum=args.momentum,
                                  eps=args.eps)

    dataset = BeliefDataset(args.dataset_root, seed=args.seed, first_k=args.first_k_data)

    train_loader = torch.utils.data.DataLoader(
            BeliefDatasetWrapper(dataset.train),
            batch_size=args.batchsize, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
            BeliefDatasetWrapper(dataset.test),
            batch_size=args.batchsize, shuffle=True, num_workers=4)

    weight_stat = common_utils.MultiCounter("./")

    for epoch in range(args.num_epoch):
        weight_stat.reset()

        weight_model.train()
        for batch in tqdm.tqdm(train_loader, total=int(len(train_loader))):
            # Train the weight model.
            obs = batch2dict(batch)
            weight_loss, stat = weight_model.loss(obs)

            for k, v in stat.items():
                weight_stat[k + "_train"].feed(v)

            weight_optim.zero_grad()
            weight_loss.backward()
            weight_g_norm = torch.nn.utils.clip_grad_norm_(
                weight_trainable_params(), args.grad_clip)
            weight_optim.step()

        # Test..
        weight_model.eval()
        for batch in tqdm.tqdm(test_loader, total=int(len(test_loader))):
            # Test the weight model.
            obs = batch2dict(batch)
            weight_loss, stat = weight_model.loss(obs)

            for k, v in stat.items(): 
                weight_stat[k + "_test"].feed(v)

        log.info("weight_stat")
        log.info(weight_stat.summary(epoch))
        weight_model.save(f"weight-{epoch}.pth")

if __name__ == "__main__":
    main()
