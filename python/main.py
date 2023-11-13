"""Run Atari Environment with DQN."""
import time
import os
import sys
import argparse
import pprint

import numpy as np
import torch

from create_envs import create_train_env, create_eval_env
import vdn_r2d2
import iql_r2d2
import common_utils
import rela
import bridge
from eval import evaluate
import utils
from utils_bridge import ActionParser


def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--method", type=str, default="iql")

    # game settings
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--greedy_extra", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=4)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-4, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--rnn_hid_dim", type=int, default=512)

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument(
        "--batchsize", type=int, default=128,
    )
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=4)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.6, help="prioritized replay alpha",
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.4, help="prioritized replay beta",
    )
    parser.add_argument("--max_len", type=int, default=40, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=50, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=1)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.4)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 10)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    game_info = utils.get_game_info("batched")

    agent = iql_r2d2.R2D2Agent(
        args.multi_step,
        args.gamma,
        0.9,
        args.train_device,
        game_info["input_dim"],
        args.rnn_hid_dim,
        game_info["num_action"],
    )
    agent_cls = iql_r2d2.R2D2Agent

    # eval is always in IQL fashion
    eval_agents = []
    for _ in range(args.num_player):
        ea = iql_r2d2.R2D2Agent(
            1,
            0.99,
            0.9,
            "cpu",
            game_info["input_dim"],
            args.rnn_hid_dim,
            game_info["num_action"],
        )
        eval_agents.append(ea)

    agent = agent.to(args.train_device)
    optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(agent)

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    ref_model = [agent_cls.clone(agent, device=args.act_device) for _ in range(3)]
    model_locker = rela.ModelLocker(ref_model, args.act_device)
    actor_eps = utils.generate_actor_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_thread
    )
    print("actor eps", actor_eps)

    actor_cons = []
    for i in range(args.num_player):
        if i % 1 == 0:
            actor_cons.append(
                lambda thread_idx: rela.R2D2Actor(
                    model_locker,
                    args.multi_step,
                    args.num_game_per_thread,
                    args.gamma,
                    args.max_len,
                    actor_eps[thread_idx],
                    1,
                    replay_buffer,
                )
            )
        else:
            actor_cons.append(
                lambda thread_idx: bridge.AllPassActor())

    context, games, actors, threads = create_train_env(
        args.method,
        args.seed,
        args.num_thread,
        args.num_game_per_thread,
        actor_cons,
        args.max_len
    )

    context.start()
    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()

    action_parser = ActionParser()

    for epoch in range(args.num_epoch):
        tachometer.start()
        stat.reset()

        action_count = None
        for batch_idx in range(args.epoch_len):
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                agent.sync_target_with_online()
            if num_update % args.actor_sync_freq == 0:
                model_locker.update_model(agent)

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            loss, priority = agent.loss(batch)
            loss = (loss * weight).mean()
            loss.backward()
            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()
            replay_buffer.update_priority(priority)

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)

            a = batch.action["a"].view(-1, 1)
            transition_size = a.size(0)
            action_hot = torch.zeros(transition_size, 39).to(a.device)
            action_hot.scatter_(1, a.long(), 1)
            if action_count is None:
              action_count = action_hot.sum(dim=0)
            else:
              action_count += action_hot.sum(dim=0)

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        tachometer.lap(
            actors, replay_buffer, args.epoch_len * args.batchsize, count_factor
        )
        stat.summary(epoch)
        #TODO, fix, since 0 is also pad
        action_sum = action_count.sum().float() - action_count[0] - action_count[38]
        print("**** action distributions ***")
        action_parser.print_distri(action_count / action_sum)

        context.pause()
        for i in range(args.num_player):
            eval_agents[i].load_state_dict(agent.state_dict())
        eval_seed = (args.seed + epoch * 1000) % 7777777
        score, perfect, _, _ = evaluate(
            eval_agents,
            "cpu",
            1000,
            eval_seed,
            0
        )
        model_saved = saver.save(agent, agent.online_net.state_dict(), score)
        print(
            "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
            % (epoch, score, perfect * 100, model_saved)
        )
        context.resume()
        print("==========")
