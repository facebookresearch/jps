import numpy as np
from copy import deepcopy

class InformationSet:
    def __init__(self, key, player_idx, num_action, explore_factor=0.01):
        # Key = all information that is available to the player.
        #       (public + the player's private information)

        self.key = key
        self.regret_sum = np.zeros(num_action)
        self.strategy_sum = np.zeros(num_action)
        self.strategy = np.repeat(1/num_action, num_action)
        self.reach_pr = 0
        self.player_idx = player_idx
        self.num_action = num_action

        self.explore_factor = explore_factor

    def next_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.strategy = self.calc_strategy()
        self.reach_pr = 0

    def calc_strategy(self):
        """
        Calculate current strategy from the sum of regret.

        ---
        Parameters

        pr: (0.0, 1.0), float
            The probability that this information set has been reached.
        """
        strategy = self.make_positive(self.regret_sum)
        total = sum(strategy)
        if total > 0:
            strategy = strategy / total
        else:
            n = self.num_action
            # Add some noise.
            strategy = np.repeat(1/n, n)

            if self.explore_factor > 0:
                strategy += np.random.randn(n) * self.explore_factor
                strategy = np.clip(strategy, 0, 1)
                strategy /= strategy.sum()

        return strategy

    def get_average_strategy(self):
        """
        Calculate average strategy over all iterations. This is the
        Nash equilibrium strategy.
        """
        total = sum(self.strategy_sum)
        if total > 0:
            strategy = self.strategy_sum / total

            # Purify
            strategy = np.where(strategy < 0.001, 0, strategy)

            # Re-normalize
            total = sum(strategy)
            strategy /= total

            return strategy

        n = self.num_action
        return np.repeat(1/n, n)

    def make_positive(self, x):
        return np.where(x > 0, x, 0)

    def __str__(self):
        strategies = ['{:03.3f}'.format(x)
                      for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)


class CFR:
    def __init__(self, game_cls, verbose=False):
        self.imap = {}
        '''
        game has the following function
        game_info = dict(public="", private=List[])
            player_idx, num_action = game.get(),
                0: chance node
                1: player1
                2: player2
            key = game.get_key()
            new_game = game.forward(action)
            terminal, utilities = game.is_terminal()
        '''
        self.game_cls = game_cls
        self.num_player = 3
        self.verbose = verbose

    def get_info_set(self, h):
        """
        Retrieve information set from dictionary.
        """
        key = h.get_key()
        if key not in self.imap:
            player_idx, num_action = h.get()
            self.imap[key] = InformationSet(key, player_idx, num_action)

        return self.imap[key]

    def cfr(self):
        init_prs = np.array([1.0, 1.0, 1.0])
        return self._cfr(self.game_cls(), prs=init_prs)

    def _cfr(self, h, prs):
        """
        Counterfactual regret minimization algorithm.

        Parameters
        ----------

        i_map: dict
            Dictionary of all information sets.
        history : [{'r', 'c', 'b'}], str
            A string representation of the game tree path we have taken.
            Each character of the string represents a single action:

            'r': random chance action
            'c': check action
            'b': bet action
        card_1 : (0, 2), int
            player A's card
        card_2 : (0, 2), int
            player B's card
        pr_1 : (0, 1.0), float
            The probability that player A reaches `history`.
        pr_2 : (0, 1.0), float
            The probability that player B reaches `history`.
        pr_c: (0, 1.0), float
            The probability contribution of chance events to reach `history`.
        """
        # print(h)
        player_idx, num_action = h.get()

        if player_idx == 0:
            # Uniform chance node.
            v = np.zeros(self.num_player)
            for i in range(num_action):
                h_next = h.forward(i)
                prs_next = deepcopy(prs)
                prs_next[player_idx] /= num_action
                v += self._cfr(h_next, prs_next)
            return v / num_action

        # Actual player.
        terminal, utilities = h.is_terminal()
        if terminal:
            return utilities

        info_set = self.get_info_set(h)
        strategy = info_set.strategy
        info_set.reach_pr += prs[player_idx]

        # Counterfactual utility per action.
        action_utils = np.zeros((self.num_player, num_action))

        for i in range(num_action):
            h_next = h.forward(i)

            prs_next = deepcopy(prs)
            prs_next[player_idx] *= strategy[i]
            action_utils[:, i] = self._cfr(h_next, prs_next)

        # Utility of information set.
        util = np.dot(action_utils, strategy)
        regrets = action_utils[player_idx, :] - util[player_idx]
        info_set.regret_sum += prs[:player_idx].prod() * prs[player_idx+1:].prod() * regrets

        return util

    def calc_next_strategy(self):
        for _, v in self.imap.items():
            v.next_strategy()

    def print_strategy(self):
        s1 = ""
        s2 = ""
        for k, v in self.imap.items():
            if v.player_idx == 1:
                s1 += str(v) + "\n"
            else:
                s2 += str(v) + "\n"

        print('player 1 strategies:')
        print(s1)
        print('player 2 strategies:')
        print(s2)

    def run(self, n_iterations):
        v = np.array([0.0, 0.0, 0.0])

        for k in range(n_iterations):
            v += self.cfr()
            self.calc_next_strategy()

            if self.verbose:
                print(f"Iteration: {k}")
                for i, vv in enumerate(v):
                    print(f'player {i} expected value: {vv / (k + 1)}')
                self.print_strategy()

        v /= n_iterations
        return v


