import torch
from torch import nn
from typing import Tuple, Dict
from .model import BridgeA2CModel, A2CModel, CommModel, BridgeA2CModel2

import torch.nn.functional as F

import sys
import os
sys.path.append("..")
import model_utils

class A2CAgent(torch.jit.ScriptModule):
    __constants__ = ["max_importance_ratio", "entropy_ratio", "min_prob", "p_hand_loss_ratio", "p_hand_loss_only_on_terminal", "explore_ratio"]

    def __init__(self, **kwargs):
        super().__init__()

        def initializer(**kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

            self.init_args = kwargs

            if self.model_type == "resnet":
                self.online_net = BridgeA2CModel(self.input_dim, self.num_action, self.hid_dim, self.num_blocks, self.opp_hand_ratio, self.use_goal, self.use_value_based, use_old_feature = self.use_old_feature)
            elif self.model_type == "e2e":
                self.online_net = BridgeA2CModel2(self.num_action,
                                                  self.embed_dim, self.hid_dim,
                                                  self.num_blocks,
                                                  self.num_rnn_layers,
                                                  self.device)
            elif self.model_type == "comm":
                self.online_net = CommModel(self.input_dim, self.hid_dim, self.num_action, self.num_blocks, self.use_value_based)
            else:
                self.online_net = A2CModel(self.input_dim, self.hid_dim, self.num_action, self.num_blocks)

            self.bce_loss = nn.BCELoss()
            self.nll_loss = nn.NLLLoss()
            self.min_prob = 1e-6

            return self

        model_utils.load_model_if_exists(kwargs, initializer)

    def clone(self, device):
        return model_utils.clone_model(self, device)

    def save(self, filename):
        model_utils.save_model(self, filename)

    def sync_target_with_online(self):
        pass
        # self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.jit.script_method
    def _act(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        reply = self.online_net.forward(obs, act=True)
        mask = obs["legal_move"].to(torch.bool)

        if "pi" in reply:
            prob = reply["pi"]
        else:
            # value-based approach. Using epsilon greedy
            adv = reply["adv"]
            max_indices = adv.max(dim=1)[1]
            prob = F.one_hot(max_indices, num_classes=adv.size(1)).float()

        explore = torch.ones_like(prob).float().to(prob.device)
        explore = explore.masked_fill(~mask, 0)
        explore = explore * self.explore_ratio / torch.sum(explore)

        prob_eps = prob * (1 - self.explore_ratio) + explore

        # Make sure no entry is smaller than zero.
        # prob = prob.clamp(min=0)
        # prob /= prob.sum(dim=1)[:, None]
        # prob = prob.clamp(min=0)
        return prob_eps.detach(), prob.detach(), reply

    @torch.jit.script_method
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prob_eps, _, reply = self._act(obs)

        # Due to a torch.Tensor.multinomial bug when input is a one_hot vector,
        # we directly assign index of 1s to action.
        # TODO: Export this as a util function.
        # v, i = prob_eps.max(1, keepdim=True)
        # mask = (v == 1)
        # action = prob_eps.multinomial(1)
        # action[mask] = i[mask]
        action = prob_eps.multinomial(1, replacement=True)

        reply["a"] = action
        reply["behavior_pi"] = prob_eps
        return reply

    @torch.jit.script_method
    def act_greedy(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        _, prob, reply = self._act(obs)
        max_action = prob.max(dim=1)[1].view(-1, 1)
        # print("getting max_action", max_action.size(), max_action)
        reply["a"] = max_action
        reply["behavior_pi"] = prob
        return reply

    @torch.jit.script_method
    def compute_priority(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        compute priority for one batch, just 1 for now.
        """
        batchsize = obs["s"].size(0)
        # print(f"Computing priority, batchsize: {batchsize}")
        priority = torch.ones(batchsize, 1)
        return { "priority" : priority }

    @torch.jit.script_method
    def loss_search(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        R = batch["R"]
        a = batch["a"]
        reply = self.online_net.forward(batch, act=False)
        V = reply["v"]

        A = R - V
        this_adv = reply["adv"].gather(1, a)

        # For policy we only use element where search = true
        # For adv/V we only use element where search = false
        search = batch["search"].squeeze(1)

        num_search = search.sum().item()
        adv_err = (this_adv[~search] - A[~search]).square().mean(0) * 0.5
        value_err = A[~search].square().mean(0) * 0.5

        err = value_err + adv_err

        stats = {
            "value_err": float(value_err.item()),
            "adv_err": float(adv_err.item()),
            "R": float(R.mean().item()),
            "V": float(V.mean().item())
        }

        if "pi" in reply:
            pi = reply["pi"].clamp(min=self.min_prob, max=1-self.min_prob)
            logpi = pi.log()
            entropy = - (pi * logpi).sum(1).mean(0)
            entropy_err = -self.entropy_ratio * entropy
            err = err + entropy_err

            stats["entropy"] = float(entropy.item())

            if num_search > 0:
                policy_err = self.nll_loss(logpi[search, :], a[search, :].squeeze())
                err = err + policy_err

                stats["policy_err"] = float(policy_err.item())

        return err, stats

    @torch.jit.script_method
    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Advantage.
        R = batch["R"]
        a = batch["a"]
        reply = self.online_net.forward(batch, act=False)
        V = reply["v"]

        A = R - V
        value_err = A.square().mean(0) * 0.5
        err = value_err

        stats = {
            "V": float(V.mean().item()),
            "R": float(R.mean().item()),
            "value_err": float(value_err.item())
        }

        if "adv" in reply:
            this_adv = reply["adv"].gather(1, a)
            adv_err = (this_adv - A).square().mean(0) * 0.5
            err = err + adv_err

            stats["adv_err"] = float(adv_err.item())

        if "pi" in reply:
            pi = reply["pi"].clamp(min=self.min_prob, max=1-self.min_prob)
            logpi = pi.log()
            entropy = - (pi * logpi).sum(1).mean(0)
            entropy_err = -self.entropy_ratio * entropy
            behavior_pi = batch["behavior_pi"].clamp(min=self.min_prob, max=1-self.min_prob)

            logpi_sel = logpi.gather(1, a)

            # Off-policy correction.
            ratio = (pi.detach() / behavior_pi).gather(1, a)
            ratio = ratio.clamp(max=self.max_importance_ratio)

            policy_err = - (logpi_sel * A.detach() * ratio).mean(dim=0)
            err = err + policy_err + entropy_err

            stats["value_err"] = float(value_err.item())
            stats["entropy"] = float(entropy.item())

        '''
        for i in range(5):
            print("S")
            print(batch["s"][i])
            print("a")
            print(batch["a"][i])
            print("R")
            print(batch["R"][i])
            print("v")
            print(batch["v"][i])
        '''

        # [TODO] In the long run we should use DDA score to guide the prediction error
        # (predict the card so that DDA achieves the smallest difference).
        if self.p_hand_loss_only_on_terminal:
            # Only care about reconstruction when the state is a terminal.
            # This might allow very long bidding sequence.
            reply["infer_p_hand"] *= batch["terminal"][:, None]
            reply["p_hand"] *= batch["terminal"][:, None]

        '''
        print(A.size())
        print(V.size())
        print(R.size())
        print(ratio.size())
        print(logpi_sel.size())
        print(policy_err.size())
        print(value_err.size())
        print(entropy.size())
        '''
        if self.p_hand_loss_ratio > 0:
            p_hand_loss = self.bce_loss(reply["infer_p_hand"], reply["p_hand"].detach())
            err += self.p_hand_loss_ratio * p_hand_loss
            stats["p_hand_loss"] = float(p_hand_loss.item())

        return err, stats
