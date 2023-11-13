import time
import torch
import common_utils
from create_envs import *


class Tachometer:
    def __init__(self):
        self.num_act = 0
        self.num_buffer = 0
        self.num_train = 0
        self.t = None
        self.total_time = 0

    def start(self, num_buffer):
        self.num_buffer = num_buffer
        self.t = time.time()

    def lap(self, actors, replay_buffer, num_train, factor):
        t = time.time() - self.t
        self.total_time += t
        # num_act = get_num_acts(actors)
        num_act = 0
        act_rate = factor * (num_act - self.num_act) / t
        num_buffer = replay_buffer.num_add()
        buffer_rate = factor * (num_buffer - self.num_buffer) / t
        train_rate = factor * num_train / t
        ret = ("Speed: train: %.1f, act: %.1f, buffer_add: %.1f, buffer_size: %d\n"
               % (train_rate, act_rate, buffer_rate, replay_buffer.size()))
        self.num_act = num_act
        self.num_buffer = num_buffer
        self.num_train += num_train
        ret += (
            "Total Time: %s, %ds\n"
            % (common_utils.sec2str(self.total_time), self.total_time)
        )
        ret += (
            "Total Sample: train: %s, act: %s\n"
            % (common_utils.num2str(self.num_train), common_utils.num2str(self.num_act))
        )
        return ret


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device).detach()
    elif isinstance(batch, dict):
        return {key: to_device(batch[key], device) for key in batch}
    elif isinstance(batch, rela.FFTransition):
        batch.obs = to_device(batch.obs, device)
        batch.action = to_device(batch.action, device)
        batch.reward = to_device(batch.reward, device)
        batch.terminal = to_device(batch.terminal, device)
        batch.bootstrap = to_device(batch.bootstrap, device)
        batch.next_obs = to_device(batch.next_obs, device)
        return batch
    elif isinstance(batch, rela.RNNTransition):
        batch.obs = to_device(batch.obs, device)
        batch.h0 = to_device(batch.h0, device)
        batch.action = to_device(batch.action, device)
        batch.reward = to_device(batch.reward, device)
        batch.terminal = to_device(batch.terminal, device)
        batch.bootstrap = to_device(batch.bootstrap, device)
        batch.seq_len = to_device(batch.seq_len, device)
        return batch
    else:
        assert False, "unsupported type: %s" % type(batch)


def get_game_info(args, factory, env_actor_gen):
    game = factory.create_train_env(0, 0)
    ea = env_actor_gen.generate(-1, -1, game, [], args, False, None, empty_init=True)

    info = {"input_dim": ea.feature_dim(game), "num_action": ea.max_num_action(game) }
    print(info)
    return info


def get_game_info_simple(factory):
    game = factory.create_train_env(0, 0)
    info = {"input_dim": game.feature_dim(), "num_action": game.max_num_action() }
    print(info)
    return info


def compute_input_dim(num_player):
    hand = 126 * num_player
    board = 76
    discard = 50
    last_action = 51 + 2 * num_player
    card_knowledge = num_player * 5 * 35
    return hand + board + discard + last_action + card_knowledge


# returns the number of steps in all actors
def get_num_acts(actors):
    total_acts = 0
    for actor in actors:
        total_acts += actor.num_act()
    return total_acts


# num_acts is the total number of acts, so total number of acts is num_acts * num_game_per_actor
# num_buffer is the total number of elements inserted into the buffer
# time elapsed is in seconds
def get_frame_stat(num_game_per_thread, time_elapsed, num_acts, num_buffer, frame_stat):
    total_sample = (num_acts - frame_stat["num_acts"]) * num_game_per_thread
    act_rate = total_sample / time_elapsed
    buffer_rate = (num_buffer - frame_stat["num_buffer"]) / time_elapsed
    frame_stat["num_acts"] = num_acts
    frame_stat["num_buffer"] = num_buffer
    return total_sample, act_rate, buffer_rate


def set_default(kwargs, key, v):
    if key not in kwargs:
        kwargs[key] = v

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)


@torch.jit.script
def get_v1(v0_joind, card_counts, ref_mask):
    v0_joind = v0_joind.cpu()
    card_counts = card_counts.cpu()

    batch, num_player, dim = v0_joind.size()
    num_player = 3
    v0_joind = v0_joind.view(batch, 1, num_player * 5, 25)

    mask = (v0_joind > 0).float()
    total_viable_cards = mask.sum()
    v1_old = v0_joind
    thres = 0.0001
    max_count = 100
    weight = 0.1
    v1_new = v1_old
    for i in range(max_count):  # can't use a variable count for tracing
        # torch.Size([256, 99, 25]) torch.Size([256, 99, 10, 25])
        # Calculate how many cards of what types are sitting in other hands.
        hand_cards = v1_old.sum(2)
        total_cards = card_counts - hand_cards
        # Exclude the cards I am holding myself.
        excluding_self = total_cards.unsqueeze(2) + v1_old
        # Negative numbers shouldn't happen, but they might (for all I know)
        excluding_self.clamp_(min=0)
        # Calculate unnormalised likelihood of cards: Adjusted count * Mask
        v1_new = excluding_self * mask
        # this is avoiding NaNs for when there are no cards.
        v1_new = v1_old * (1 - weight) + weight * v1_new
        v1_new = v1_new / (v1_new.sum(-1, keepdim=True) + 1e-8)
        # if False: # this is strictly for debugging / diagnostics
        #     # Normalise the diff by total viable cards.
        #     diff = (v1_new - v1_old).abs().sum() / total_viable_cards
        #     xent = get_xent(data, v1_old[:,:,:5,:])
        #     print('diff %8.3g  xent %8.3g' % (diff, xent))
        v1_old = v1_new

    return v1_new

@torch.jit.script
def check_v1(v0, v1, card_counts, mask):
    ref_v1 = get_v1(v0, card_counts, mask)
    batch, num_player, dim = v1.size()
    # print('v1:', v1.size())
    # print('v0:', v0.size())
    # print('ref_v1:', ref_v1.size())
    v1 = v1.view(batch, 1, 3 * 5, 25).cpu()
    # print('v1:', v1.size())
    # print('ref_v1:', ref_v1.size())
    print("diff: ", (ref_v1 - v1).max())
    if (ref_v1 - v1).max() > 1e-4:
        print((ref_v1 - v1)[0][0][0])
        assert False


def check_trajectory(batch):
    assert batch.obs["h"][0][0].sum() == 0
    length = batch.obs["h"][0].size(0)
    end = 0
    for i in range(length):
        t = batch.terminal[0][i]

        if end != 0:
            assert t

        if not t:
            continue

        if end == 0:
            end = i
    print("trajectory ends at:", end)
