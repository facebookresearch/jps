import torch
from torch import nn
from typing import Tuple, Dict
import torch.nn.functional as F

import sys
sys.path.append("..")
from model_utils import Swish, LinearList


def post_process(obs: Dict[str, torch.Tensor], reply: Dict[str, torch.Tensor], verbose: bool=False) -> None:
    mask = obs["legal_move"].to(torch.bool)

    if "adv" in reply:
        reply["adv"].masked_fill_(~mask, -1e38)

    # use different policy head depending on which player is playing.
    if "pi" in reply:
        policy = reply["pi"]

        if verbose:
            print("Raw policy")
            print(policy[:5])

        policy = policy.masked_fill(~mask, 0)
        policy_sum = policy.sum(dim=1)

        zero_sel = policy_sum < 1e-5
        policy[zero_sel, :] = mask[zero_sel, :].float()
        policy_sum[zero_sel] = policy[zero_sel, :].sum(dim=1)
        policy = policy / policy_sum[:, None]

        if verbose:
            print("Policy:")
            print(policy[:5])

        reply["pi"] = policy

    if verbose:
        # print(reply.keys())
        print("Value:")
        print(reply["v"][:5])


class A2CModel(torch.jit.ScriptModule):
    __constants__ = ["hid_dim", "num_lstm_layer", "in_dim"]

    def __init__(self, in_dim, hid_dim, num_action, num_layer):
        super().__init__()
        self.in_dim = in_dim
        self.num_action = num_action
        self.hid_dim = hid_dim
        self.num_layer = num_layer

        in_dim = self.in_dim
        nets = []
        for i in range(self.num_layer):
            nets.append(nn.Linear(in_dim, self.hid_dim))
            nets.append(nn.ReLU())
            in_dim = self.hid_dim

        self.net = nn.Sequential(*nets)

        self.fc_v = nn.Linear(self.hid_dim, 1)
        self.fc_a = nn.Linear(self.hid_dim, self.num_action)
        self.softmax = nn.Softmax(dim=1)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor], act : bool) -> Dict[str, torch.Tensor]:
        # print(obs.keys())
        # print(f"Forwarding with batchsize: {obs['s'].size(0)}")
        s = obs["s"][:, :self.in_dim]

        assert s.dim() == 2, "should be 2 [batch, dim], get %d" % s.dim()
        x = self.net(s)
        pi = self.softmax(self.fc_a(x))
        v = self.fc_v(x)
        result = {"pi": pi, "v": v}
        # print(result)
        return result


class BridgeA2CModel(torch.jit.ScriptModule):
    __constants__ = ["input_dim", "num_action", "pass_action", "hidden_dim", "train_hand", "hand_dim", "use_opp_hand", "use_goal", "use_value_based"]

    def __init__(self, input_dim, num_action, hidden_dim, num_blocks, opp_hand_ratio, use_goal, use_value_based, use_old_feature = False):
        super().__init__()
        self.num_action = num_action
        self.pass_action = num_action - 3
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.opp_hand_ratio = opp_hand_ratio
        self.train_hand = False
        self.hand_dim = 52
        self.use_value_based = use_value_based
        self.input_dim = input_dim if not use_old_feature else 229

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.enn_partner = LinearList(self.hidden_dim, self.num_blocks)

        self.use_opp_hand = False
        if self.opp_hand_ratio > 0:
            self.use_opp_hand = True
            self.enn_opp = LinearList(self.hidden_dim, self.num_blocks)
        self.linear2 = nn.Linear(self.hidden_dim, self.hand_dim)

        self.linear3 = nn.Linear(self.input_dim + self.hand_dim, self.hidden_dim)
        self.pnn = LinearList(self.hidden_dim, self.num_blocks)

        self.linear_policy = nn.Linear(self.hidden_dim, self.num_action)
        self.linear_goal = nn.Linear(self.hidden_dim, self.pass_action)
        # self.support = torch.Tensor(range(self.pass_action))
        self.linear_value = nn.Linear(self.hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.use_goal = use_goal

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor], act: bool) -> Dict[str, torch.Tensor]:
        # print(obs.keys())
        # if "s" not in obs:
        #    print(obs.keys())
        #    print(obs)

        s = obs["s"][:, :self.input_dim]

        # print("State:")
        # print(s[0])

        # x = x[:, :, 0]
        # print('>>>', x.size())
        # print(x.max(), x.mean())
        # assert x.max() <= 1.0
        # print(x.mean())
        # x.normal_()
        batchsize = s.size(0)
        # s = x.narrow(1, 0, self.input_dim)

        x1 = self.linear1(s)
        x1_p = self.enn_partner(x1)
        infer_p_hand = self.linear2(x1_p)
        infer_p_hand = infer_p_hand.sigmoid()

        x2 = torch.cat((s, infer_p_hand), dim=1)
        x2 = self.linear3(x2)
        h1 = self.pnn(x2)
        h2 = self.linear_policy(h1)

        value = self.linear_value(h1)
        reply = { "v": value }

        if self.use_value_based:
            # Treat it as advantage function
            reply["adv"] = h2
        else:
            reply["pi"] = self.softmax(h2)

        post_process(obs, reply)

        # print(reply.keys())
        # print("Policy:")
        # print(policy[0])

        if act:
            return reply

        if "p_hand" in obs:
            p_hand = obs["p_hand"]
            # Only predict AKQJ for each suit
            p_hand_hcp = obs["p_hand_hcp"]

            reply["p_hand"] = (p_hand * p_hand_hcp).float()
            reply["infer_p_hand"] = (infer_p_hand * p_hand_hcp)

        # print("p_hand; ", p_hand.size())
        # print("p_hand_hcp: ", p_hand_hcp.size())
        # print("infer_p_hand: ", infer_p_hand.size())


        '''
        if self.use_opp_hand:
            x1_o = self.enn_partner(x1)
            infer_o_hand = self.linear2(x1_o)
            infer_o_hand = infer_o_hand.sigmoid()
            o_hand_gt = x.narrow(1, self.input_dim + self.num_action + self.hand_dim, self.hand_dim)
            o_hand_loss = self.bce_loss(infer_o_hand, o_hand_gt.float())

            reply["o_hand_loss"] = o_hand_loss
        '''

        return reply
        #import pdb; pdb.set_trace()


class CommModel(torch.jit.ScriptModule):
    __constants__ = ["input_dim", "num_action", "hidden_dim", "use_value_based"]

    def __init__(self, input_dim, hidden_dim, num_action, num_blocks, use_value_based):
        super().__init__()
        self.input_dim = input_dim
        self.num_action = num_action
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.use_value_based = use_value_based

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden = LinearList(self.hidden_dim, self.num_blocks)

        # Player 1 and player 2 head.
        self.linear_policy1 = nn.Linear(self.hidden_dim, self.num_action)
        self.linear_policy2 = nn.Linear(self.hidden_dim, self.num_action)
        self.linear_value = nn.Linear(self.hidden_dim, 1)

        # A head.
        self.linear_A = nn.Linear(self.hidden_dim, self.num_action)

        self.softmax = nn.Softmax(dim=1)

    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor], act: bool) -> Dict[str, torch.Tensor]:
        # print(obs.keys())
        if "s" not in obs:
            print(obs.keys())
            print(obs)

        s = obs["s"][:, :self.input_dim]

        verbose = False

        if verbose:
            print("State:")
            print(s[:5])
            print("PlayerId:")
            print(obs["player_idx"][:5])

        batchsize = s.size(0)

        h = F.relu(self.linear1(s))
        h = self.hidden(h)

        adv = self.linear_A(h)
        value = self.linear_value(h)

        reply = {
            "v": value,
            "adv": adv
        }

        # use different policy head depending on which player is playing.
        if not self.use_value_based:
            h1 = self.linear_policy1(h)
            h2 = self.linear_policy2(h)
            # hp = torch.stack([policy1, policy2]).gather(0, obs["player_idx"].unsqueeze(0).unsqueeze(2))
            hp = (1 - obs["player_idx"]) * h1 + obs["player_idx"] * h2
            # hp = h1
            policy = self.softmax(hp)
            reply["pi"] = policy + 1e-6

        post_process(obs, reply, verbose)

        return reply



class BridgeA2CModel2(torch.jit.ScriptModule):
    def __init__(self,
                 num_action,
                 embed_dim,
                 hidden_dim,
                 num_blocks,
                 num_rnn_layers,
                 device=None):
        super().__init__()

        self.num_action = num_action
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_rnn_layers = num_rnn_layers

        self.device = device

        self.deck_size = 52
        self.num_players = 4
        self.max_bid_len = 40
        self.num_bids = 39
        self.num_tricks = 13

        self.vul_dim = 2
        self.stage_dim = 2
        # self.bid_seq_dim = self.max_bid_len * self.embed_dim
        self.doubled_dim = 2
        self.s_dim = self.num_players * self.deck_size
        self.trick_dim = self.num_players * self.embed_dim
        # self.play_card_dim = self.deck_size * self.embed_dim
        # self.play_seat_dim = self.deck_size * self.num_players
        # self.win_dim = self.num_tricks * self.num_players

        self.input_dim = self.vul_dim + self.stage_dim + self.hidden_dim + \
                self.embed_dim + self.doubled_dim + self.num_players + \
                self.s_dim + self.trick_dim + self.hidden_dim + self.num_action

        self.bid_embed = nn.Embedding(self.num_bids,
                                      self.embed_dim,
                                      padding_idx=self.num_bids - 1)
        self.card_embed = nn.Embedding(self.deck_size + 1,
                                       self.embed_dim,
                                       padding_idx=self.deck_size)
        self.swish = Swish()

        self.linear_b = nn.Linear(self.embed_dim + self.num_players,
                                  self.hidden_dim)
        self.linear_p = nn.Linear(
            self.embed_dim + self.embed_dim + self.num_players,
            self.hidden_dim)

        self.rnn_b = nn.LSTM(self.hidden_dim,
                             self.hidden_dim,
                             num_layers=self.num_rnn_layers)
        self.rnn_p = nn.LSTM(self.hidden_dim,
                             self.hidden_dim,
                             num_layers=self.num_rnn_layers)
        if self.device is not None:
            # Because of a torchscript issue for rnn, we have to put rnn to the
            # desired device first.
            self.rnn_b = self.rnn_b.to(self.device)
            self.rnn_p = self.rnn_p.to(self.device)

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden1 = LinearList(self.hidden_dim,
                                  self.num_blocks,
                                  act="swish")

        self.linear2 = nn.Linear(self.input_dim, self.hidden_dim)
        self.hidden2 = LinearList(self.hidden_dim,
                                  self.num_blocks,
                                  act="swish")

        self.linear_value = nn.Linear(self.hidden_dim, 1)
        self.linear_policy = nn.Linear(self.hidden_dim, self.num_action)

        self.softmax = nn.Softmax(dim=1)

        self.rnn_b.flatten_parameters()
        self.rnn_p.flatten_parameters()

    @torch.jit.script_method
    def _seat_to_one_hot(self, seat: torch.Tensor, padding_idx: int):
        mask = (seat == padding_idx)
        result = seat.masked_fill(mask, 0)
        result = F.one_hot(result, self.num_players)
        result.masked_fill_(mask.unsqueeze(-1), 0)
        return result

    @torch.jit.script_method
    def _extract_bidding_feature(self, bid: torch.Tensor):
        batch_size = bid.size(0)

        bid_b = self.bid_embed(bid[:, :, 0])
        bid_s = bid[:, :, 1]  # [batch_size, seq_len]
        # mask = (bid_s == -1).transpose(0, 1).contiguous()
        mask = (bid_s != -1).transpose(0, 1).contiguous()

        bid_s = self._seat_to_one_hot(bid_s, -1).float()
        bid_seq = torch.cat([bid_b, bid_s], dim=-1).transpose(0, 1).contiguous()
        # [seq_len, batch_size, embed_dim]
        bid_seq = self.linear_b(bid_seq)
        bid_seq = self.swish(bid_seq)
        bid_h0 = torch.zeros(self.num_rnn_layers,
                             batch_size,
                             self.hidden_dim,
                             dtype=bid_seq.dtype,
                             device=bid_seq.device)
        bid_c0 = torch.zeros(self.num_rnn_layers,
                             batch_size,
                             self.hidden_dim,
                             dtype=bid_seq.dtype,
                             device=bid_seq.device)

        output, (_, _) = self.rnn_b(bid_seq, (bid_h0, bid_c0))
        # # [seq_len, batch_size, hid_dim]
        # bid_feature = output.masked_fill(mask.unsqueeze(-1), 0)
        # bid_feature = output.sum(dim=0)  # [batch_size, hid_dim]

        output = torch.cat([bid_h0[-1, :].unsqueeze(0), output], dim=0)
        index = mask.long().sum(dim=0, keepdim=True).unsqueeze(-1).expand(
            1, batch_size, self.hidden_dim)  # [1, batch_size, hid_dim]
        bid_feature = output.gather(0, index).squeeze(0)

        return bid_feature

    @torch.jit.script_method
    def _extract_playing_feature(self, play: torch.Tensor,
                                 contract: torch.Tensor):
        batch_size = play.size(0)

        c = contract.unsqueeze(1).expand(-1, self.deck_size, -1)
        play_c = self.card_embed(play[:, :, 0])
        play_s = play[:, :, 1]  # [batch_size, seq_len]
        # mask = (play_s == -1).transpose(0, 1).contiguous()
        mask = (play_s != -1).transpose(0, 1).contiguous()

        play_s = self._seat_to_one_hot(play_s, -1).float()
        play_seq = torch.cat([c, play_c, play_s],
                             dim=-1).transpose(0, 1).contiguous()
        # [seq_len, batch_size, embed_dim]
        play_seq = self.linear_p(play_seq)
        play_seq = self.swish(play_seq)
        play_h0 = torch.zeros(self.num_rnn_layers,
                              batch_size,
                              self.hidden_dim,
                              dtype=play_seq.dtype,
                              device=play_seq.device)
        play_c0 = torch.zeros(self.num_rnn_layers,
                              batch_size,
                              self.hidden_dim,
                              dtype=play_seq.dtype,
                              device=play_seq.device)

        output, (_, _) = self.rnn_p(play_seq, (play_h0, play_c0))
        # # [seq_len, batch_size, hid_dim]
        # play_feature = output.masked_fill(mask.unsqueeze(-1), 0)
        # play_feature = output.sum(dim=0)  # [batch_size, hid_dim]

        output = torch.cat([play_h0[-1, :].unsqueeze(0), output], dim=0)
        index = mask.long().sum(dim=0, keepdim=True).unsqueeze(-1).expand(
            1, batch_size, self.hidden_dim)  # [1, batch_size, hid_dim]
        play_feature = output.gather(0, index).squeeze(0)

        return play_feature



    @torch.jit.script_method
    def forward(self, obs: Dict[str, torch.Tensor],
                act: bool) -> Dict[str, torch.Tensor]:

        vul = obs["vul"]
        stage = obs["stage"]
        bid = obs["bid"]
        contract = obs["contract"]
        doubled = obs["doubled"]
        declarer = obs["declarer"]
        s = obs["s"]
        trick = obs["trick"]
        play = obs["play"]
        legal_move = obs["legal_move"]

        stage = F.one_hot(stage, self.stage_dim).float().squeeze_(1)
        contract = self.bid_embed(contract).squeeze_(1)
        declarer = self._seat_to_one_hot(declarer, -1).float().squeeze_(1)
        s = s.view(-1, self.s_dim)
        trick = self.card_embed(trick).view(-1, self.trick_dim)
        bid = self._extract_bidding_feature(bid)
        play = self._extract_playing_feature(play, contract)

        x = torch.cat([
            vul, stage, bid, contract, doubled, declarer, s, trick, play,
            legal_move
        ],
                      dim=1)

        h1 = self.linear1(x)
        # h = F.relu(h)
        h1 = self.swish(h1)
        h1 = self.hidden1(h1)

        h2 = self.linear2(x)
        # h2 = F.relu(h2)
        h2 = self.swish(h2)
        h2 = self.hidden2(h2)

        v = self.linear_value(h1)
        v = torch.tanh(v)

        pi = self.linear_policy(h2)
        pi = self.softmax(pi)

        reply = {"pi": pi, "v": v}
        post_process(obs, reply)

        return reply
