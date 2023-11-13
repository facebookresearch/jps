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

@hydra.main(config_path="conf/partition_train.yaml", strict=True)
def main(args):
    # torch.backends.cudnn.benchmark = True
    log.info(common_utils.init_context(args))
    factory = hydra.utils.instantiate(args.game)
    factory.set_args(args)

    game_info = utils.get_game_info_simple(factory)

    print("Create weight predict model")
    args.weight_model.params.input_dim = game_info["input_dim"]

    assert args.weight_model.params.input_dim == 1653

    model = hydra.utils.instantiate(args.weight_model)
    model = model.to(args.train_device)
    trainable_params = lambda: model.parameters()

    optim = make_optimizer(trainable_params(),
                           optim=args.optim,
                           lr=args.lr,
                           momentum=args.momentum,
                           eps=args.eps)

    dataset = BeliefDataset(args.dataset_root, seed=args.seed, first_k=args.first_k_data)

    train_loader = torch.utils.data.DataLoader(
            BeliefDatasetWrapper(dataset.train),
            batch_size=args.batchsize, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
            BeliefDatasetWrapper(dataset.test),
            batch_size=args.batchsize, shuffle=True, num_workers=4)

    stats = common_utils.MultiCounter("./")

    for epoch in range(args.num_epoch):
        stats.reset()

        model.train()
        for batch in tqdm.tqdm(train_loader, total=int(len(train_loader))):
            # Train the weight model.
            obs = batch2dict(batch)
            loss, stat = model.loss(obs)

            for k, v in stat.items():
                stats[k + "_train"].feed(v)

            optim.zero_grad()
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params(), args.grad_clip)
            optim.step()

        # Test..
        model.eval()
        for batch in tqdm.tqdm(test_loader, total=int(len(test_loader))):
            # Test the weight model.
            obs = batch2dict(batch)
            loss, stat = model.loss(obs)

            for k, v in stat.items(): 
                stats[k + "_test"].feed(v)

        log.info("stats")
        log.info(stats.summary(epoch))
        model.save(f"weight_pred-{epoch}.pth")

if __name__ == "__main__":
    main()
