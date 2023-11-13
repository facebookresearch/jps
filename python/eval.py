import time
import numpy as np
import torch

from create_envs import create_eval_env
import rela
import iql_r2d2
import utils
import baseline_model
import bridge


def evaluate(
    model,
    device,
    num_game,
    seed,
    eval_eps,
    log_prefix=None,
):
    actor_cons = []
    for i in range(4):
        if i % 2 == 0:
            model_locker = rela.ModelLocker([model[0]], device)
            cons = lambda: rela.R2D2Actor(model_locker, 1, eval_eps)
            #cons = lambda: bridge.AllPassActor()
        else:
            baseline = baseline_model.BaselineBridgeModel()
            cons = lambda: bridge.BaselineActor(baseline, device)
            #cons = lambda: bridge.AllPassActor()
        actor_cons.append(cons)

    context, games = create_eval_env(
        seed,
        num_game,
        actor_cons,
        eval_eps,
        log_prefix,
    )
    context.start()
    while not context.terminated():
        time.sleep(0.5)

    context.terminate()
    while not context.terminated():
        time.sleep(0.5)
    scores = [g.get_episode_reward() for g in games]
    num_perfect = np.sum([1 for s in scores if s == 25])
    return np.mean(scores), num_perfect / len(scores), scores, num_perfect


def get_num_player(filename):
    if "NUM_PLAYER2" in filename:
        return 2
    if "NUM_PLAYER3" in filename:
        return 3
    if "NUM_PLAYER4" in filename:
        return 4
    if "NUM_PLAYER5" in filename:
        return 5
    print("Warning: cannot finding #player, revert to 2")
    return 2


def evaluate_saved_model(
    weight_file, device, num_game, seed, bomb, num_run=1, log_prefix=None,
):
    print("evaluating: %s \n\t for %dx%d games" % (weight_file, num_run, num_game))
    if "GREEDY_EXTRA1" in weight_file:
        greedy_extra = 1
    else:
        greedy_extra = 0

    device = "cpu"
    num_player = get_num_player(weight_file)
    game_info = utils.get_game_info(num_player, greedy_extra)
    input_dim = game_info["input_dim"]
    output_dim = game_info["num_action"]
    hid_dim = 512

    actor = iql_r2d2.R2D2Agent(1, 0.99, 0.9, device, input_dim, hid_dim, output_dim)
    state_dict = torch.load(weight_file)
    if "pred.weight" in state_dict:
        state_dict.pop("pred.bias")
        state_dict.pop("pred.weight")

    actor.online_net.load_state_dict(state_dict)

    scores = []
    perfect = 0
    for i in range(num_run):
        _, _, score, p = evaluate(
            actor,
            device,
            num_game,
            num_game * i + seed,
            0,
        )
        scores.extend(score)
        perfect += p

    mean = np.mean(scores)
    print(len(scores))
    sem = np.std(scores) / np.sqrt(len(scores))
    perfect_rate = perfect / (num_game * num_run)
    print("score: %f +/- %f" % (mean, sem), "; perfect: ", perfect_rate)
    return mean, sem, perfect_rate


if __name__ == "__main__":
    import os
    import sys
    import common_utils

    save_dir = "exps/test_eval"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger_path = os.path.join(save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)

    #weight_file = "/private/home/hengyuan/hanabi-joint/pyhanabi/sweep/r2d2_method_v0/NUM_PLAYER2_TRAIN_BOMB0_EVAL_BOMB0_FIXED_EPS1_GREEDY_EXTRA1_SEED17779999/model4.pthw"
    num_game = 1000
    seed = 1
    #evaluate_saved_model(weight_file, "cpu", num_game, seed, 0)
    game_info = utils.get_game_info()
    input_dim = game_info["input_dim"]
    output_dim = game_info["num_action"]
    hid_dim = 200
    model = iql_r2d2.R2D2Agent(1, 0.99, 0.9, "cpu", input_dim, hid_dim, output_dim)

    a ,b, c, d = evaluate([model], "cpu", 1, 1, 1)
    print(a)
