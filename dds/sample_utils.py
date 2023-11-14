import card_utils
import ctypes
import dds
import random
import redeal
import itertools

import numpy as np
import pandas as pd

from tabulate import tabulate


class DealWalk:
    def __init__(self, deal=None):
        # For each card, specify which player owns it.
        if deal is None:
            self.card_map = card_utils.make_random_card_map()
        else:
            self.card_map = card_utils.make_card_map(deal)

    @classmethod
    def from_card_map(cls, card_map):
        '''
           Input card_map:
               52 dim, 13C 13D 13H 13S (CA = 0), can also be list of list (4x13)
               For each dim, there is assignment NESW (N=0)
        '''
        self = cls()
        if len(card_map) == 4:
            card_map = list(itertools.chain.from_iterable(card_map))
        assert len(card_map) == card_utils.NUM_CARDS

        # Check whether card_map is valid.
        counts = [0] * card_utils.NUM_PLAYERS
        for v in card_map:
            counts[v] += 1

        for i in range(card_utils.NUM_PLAYERS):
            # Ensure every player has 13 cards.
            assert counts[i] == card_utils.NUM_CARD_PER_PLAYER, f"CardMap not valid: player {card_utils.IDX2SEATS[i]} has {counts[i]} cards!"

        self.card_map = card_map
        return self

    @classmethod
    def from_pbn(cls, pbn):
        self = cls()
        self.card_map = card_utils.make_card_map(pbn)
        return self

    def tolist(self):
        deal = []
        for i in range(4):
            deal.append([[] for j in range(4)])

        for i, p in enumerate(self.card_map):
            # Note that card_utils use CDHS suit order (same as the c++ code).
            # But the output deal should be SHDC format.
            deal[p][card_utils.suit_swap(
                i // card_utils.NUM_CARD_PER_SUIT)].append(card_utils.rank(i))

        return deal

    def get_pbn_noprefix(self):
        deal = self.tolist()

        ret = []
        for p in deal:
            suits = []
            for s in p:
                suits.append("".join(s))

            ret.append(".".join(suits))

        return " ".join(ret)

    def get_pbn(self):
        return "N:" + self.get_pbn_noprefix()

    def get_hcp(self):
        pbn = self.get_pbn_noprefix()
        hcp = dict(A=4,K=3,Q=2,J=1)
        return [ sum([ hcp.get(c, 0) for c in hands ]) for hands in pbn.split(" ") ]

    def get_card_map_mask(self, suits_filter):
        '''
            suits_filter: List[List[Bool]]: 4 player (NESW) x 4 suits(SHDC order)
            If suits_filters['N']['C'] == True, then all North's Club suits are masked as True, etc.
        '''
        mask = [False] * card_utils.NUM_CARDS
        for i in range(len(mask)):
            p = self.card_map[i]
            suit_idx = card_utils.suit_swap(i // card_utils.NUM_CARD_PER_SUIT)
            mask[i] = suits_filter[p][suit_idx]

        return mask

    def resample_with_mask(self, mask, suits_in_lock=None, iter=10):
        '''
            Resample cards while keeping cards with mask = 1 at fixed owner.
            suits_in_lock: List[List[Bool]]: 4 player (NESW) x 4 suits(SHDC order)
            If suits_in_lock['N']['C'] == True, then North won't get any more Clubs than those fixed by the masks.
        '''
        movable = []
        for i, m in enumerate(mask):
            if not m:
                movable.append(i)

        for t in range(iter):
            random.shuffle(movable)
            found = False
            # Find a pair so that they can swap.
            for i in range(len(movable)):
                card_i = movable[i]
                suit_i = card_utils.suit_swap(card_i // card_utils.NUM_CARD_PER_SUIT)
                pi = self.card_map[card_i]

                for j in range(i):
                    card_j = movable[j]
                    suit_j = card_utils.suit_swap(card_j // card_utils.NUM_CARD_PER_SUIT)
                    pj = self.card_map[card_j]

                    if pi != pj and not suits_in_lock[pi][suit_j] and not suits_in_lock[pj][suit_i]:
                        # Found one.
                        found = True
                        break

                if found:
                    break

            if not found:
                return False

            # Swap card_i and card_j
            self.card_map[card_i] = pj
            self.card_map[card_j] = pi
            # print(f"Swapping {card_utils.idx2card(card_i)} and {card_utils.idx2card(card_j)} between {card_utils.IDX2SEATS[pi]} and {card_utils.IDX2SEATS[pj]}")

        return True

    def shuffle3(self, fix_player=0):
        '''Fix player and shuffle the other three cards'''
        other_cards = [
            i for i, player in enumerate(self.card_map) if player != fix_player
        ]
        random.shuffle(other_cards)
        # Then we reassign their owner.
        i = 0
        for player in range(4):
            if player == fix_player:
                continue
            b = i * 13
            e = (i + 1) * 13
            for j in range(b, e):
                self.card_map[other_cards[j]] = player

            i += 1

    def swap2cards3(self, fix_player=0):
        '''
        One random step: randomly swap card from one to the other player.
        When picking players, never pick fix_player.
        '''
        # Pick two card and swap.
        cards = [
            i for i, player in enumerate(self.card_map) if player != fix_player
        ]
        random.shuffle(cards)

        i = 1
        while self.card_map[cards[0]] == self.card_map[cards[i]]:
            i += 1

        # now swap
        self.card_map[cards[0]], self.card_map[cards[i]] = self.card_map[
            cards[i]], self.card_map[cards[0]]

    def swap2cards(self):
        ''' One random step: randomly swap card from one to the other '''
        # Pick two card and swap.
        cards = list(range(52))
        random.shuffle(cards)

        i = 1
        while self.card_map[cards[0]] == self.card_map[cards[i]]:
            i += 1

        # now swap
        self.card_map[cards[0]], self.card_map[cards[i]] = self.card_map[
            cards[i]], self.card_map[cards[0]]

    def fill_dds_deal(self, deal: dds.ddTableDeal):
        ''' Fill in structure '''
        for i in range(4):
            for j in range(4):
                deal.cards[i][j] = 0

        for i, p in enumerate(self.card_map):
            deal.cards[p][card_utils.suit_swap(
                i //
                card_utils.NUM_CARD_PER_SUIT)] += card_utils.dds_bit_rank(i)

    def redeal_deal(self):
        ''' Convert the current deal to redeal.Deal '''
        deal = self.tolist()
        predeal = dict()
        for p, key in zip(deal, "NESW"):
            suits = []
            for s in p:
                if len(s) == 0:
                    suits.append("-")
                else:
                    suits.append("".join(s))

            predeal[key] = " ".join(suits)

        dealer = redeal.Deal.prepare(predeal)
        return dealer()


def compute_dd_table_backup(ws):
    results = np.zeros((len(ws), 4, 5))
    for k, w in enumerate(ws):
        deal = w.redeal_deal()
        # The output is automatically converted to CDHSN
        results[k, :, :] = np.array(deal.dd_table())

    return results


def compute_dd_table(ws):
    table_deals = dds.ddTableDeals()
    table_res = dds.ddTablesRes()
    allParResults = dds.allParResults()

    table_deals.noOfTables = len(ws)

    # Fill in deal
    for i, w in enumerate(ws):
        w.fill_dds_deal(table_deals.deals[i])

    # By default they initialized to zero.
    trumpFilter = ctypes.c_int * 5

    ret_val = dds.CalcAllTables(ctypes.pointer(table_deals), 0, trumpFilter(),
                                ctypes.pointer(table_res),
                                ctypes.pointer(allParResults))
    if ret_val < 0:
        print(f"Something wrong with ret_val = {ret_val}")

    assert table_res.noOfBoards == len(
        ws) * 5 * 4, f"{table_res.noOfBoards} != {len(ws)} * 5 * 4"

    # Read results.
    results = np.zeros((len(ws), 4, 5), dtype=np.int)
    for k in range(len(ws)):
        res = table_res.results[k]
        for i in range(5):
            # SHDCN to CDHSN
            idx = card_utils.strain_swap(i)
            # NESW
            for j in range(4):
                results[k, j, idx] = res.resTable[i][j]

    return results


def print_dd_table(tbl):
    res = "     C   D   H   S   N \n"
    if isinstance(tbl, np.ndarray):
        # tbl: 4 x 5 np.array
        for j, p in enumerate("NESW"):
            res += f"{p} "
            for i, s in enumerate("CDHSN"):
                res += " %3d" % int(tbl[j, i])
            res += "\n"
    elif isinstance(tbl, list):
        # tbl: column first list
        for j, p in enumerate("NESW"):
            res += f"{p} "
            for i, s in enumerate("CDHSN"):
                res += " %3d" % tbl[i * 4 + j]
            res += "\n"

    print(res)

def print_bidding_table(dealer, bidd):
    '''
        dealer: 0 - 3 (NESW)
        bidd: list of string
    '''
    s = ""
    for a in "NESW":
        s += "  " + a + "  "

    s += "\n"
    for _ in range(dealer):
        s += " " * 5

    for b in bidd:
        if b[0] == '(':
            b = b[1:-1]
        s += f"  {b:2} "
        dealer += 1
        if dealer >= 4:
            s += "\n"
            dealer = 0

    if dealer > 0:
        s += "\n"
    print(s)

def get_future_tricks(deals, strain, first):
    '''
        deals is List[DealWalk]
        strain: the contract strain, can be a list or a number.
        first: the first player to play the game, can be a list or a number.
    '''

    if not isinstance(strain, list):
        strain = [strain] * len(deals)

    if not isinstance(first, list):
        first = [first] * len(deals)

    bo = dds.boardsPBN()
    solved = dds.solvedBoards()
    line = ctypes.create_string_buffer(80)

    def get_card(suit, rank):
        # Convert dds format to CardUtil format.
        # Suit: card_utils uses CDHS (same as the c++ code) while dds uses SHDC.
        # rank: card_utils uses A=0, K=1, while dds use A=14, K=13
        return card_utils.card2idx(card_utils.suit_swap(suit),
                                   card_utils.dds_rank2rank(rank))

    num_simulations = len(deals)
    future_tricks = np.zeros((num_simulations, card_utils.NUM_CARDS))
    mean_tricks = np.zeros(num_simulations)

    # For cards not belongs to the first player, its value will be -1
    future_tricks[:] = -1

    start = 0
    max_bs = 200

    while start < num_simulations:
        n = min(num_simulations - start, max_bs)

        for i in range(n):
            idx = i + start

            # Convert from our convention to dds convention.
            bo.deals[i].trump = card_utils.strain_swap(strain[idx])
            bo.deals[i].first = first[idx]

            bo.deals[i].currentTrickSuit[0] = 0
            bo.deals[i].currentTrickSuit[1] = 0
            bo.deals[i].currentTrickSuit[2] = 0

            bo.deals[i].currentTrickRank[0] = 0
            bo.deals[i].currentTrickRank[1] = 0
            bo.deals[i].currentTrickRank[2] = 0

            bo.deals[i].remainCards = bytes(deals[idx].get_pbn(), 'utf8')

            bo.target[i] = -1
            bo.solutions[i] = 3
            bo.mode[i] = 0

        bo.noOfBoards = n
        res = dds.SolveAllBoards(ctypes.pointer(bo), ctypes.pointer(solved))

        if res != dds.RETURN_NO_FAULT:
            dds.ErrorMessage(res, line)
            print("DDS error {}".format(line.value.decode("utf-8")))

        for i in range(n):
            idx = i + start

            fut = solved.solvedBoards[i]
            # used to normalize trick scores
            score_sum = 0
            score_num = 0
            for j in range(fut.cards):
                card = get_card(fut.suit[j], fut.rank[j])

                future_tricks[idx][card] = fut.score[j]
                score_sum += fut.score[j]
                score_num += 1
                #equals is in bitwise holding encoding
                equals = fut.equals[j]
                rank = 0
                while equals > 0:
                    if equals % 2 == 1:
                        card = get_card(fut.suit[j], rank)
                        future_tricks[idx][card] = fut.score[j]
                        score_sum += fut.score[j]
                        score_num += 1
                    equals >>= 1
                    rank += 1
            mean_tricks[idx] = score_sum * 1.0 / score_num

        start += n

    return future_tricks, mean_tricks


def print_simulation(future_tricks_gt, future_tricks=None):
    '''
       future_tricks_gt: the ground truth future trick, 52 dim     (type: np.array or list)
       future_tricks:    the sampled future trick,      N x 52 dim (type: np.array)
    '''
    if isinstance(future_tricks_gt, list):
        future_tricks_gt = np.array(future_tricks_gt)

    if future_tricks is None:
        avg_tricks = None
        headers = ["card", "gt_trick"]
    else:
        avg_tricks = np.mean(future_tricks, axis=0)
        max_tricks = np.max(future_tricks, axis=0)
        argmax_tricks = np.argmax(future_tricks, axis=0)

        min_tricks = np.min(future_tricks, axis=0)
        argmin_tricks = np.argmin(future_tricks, axis=0)

        dist_sqr = (future_tricks - future_tricks_gt[None,:]) ** 2
        min_dists = np.min(dist_sqr, axis=0)
        argmin_dists = np.argmin(dist_sqr, axis=0)

        headers = ["card", "gt_trick", "avg", "min", "argmin", "max", "argmax", "mindist", "argmin_dist"]

    if isinstance(future_tricks_gt, list):
        future_tricks_gt = np.array(future_tricks_gt)

    tbl = []

    for i in range(future_tricks_gt.shape[0]):
        v0 = future_tricks_gt[i]
        if v0 < 0:
            continue

        card = card_utils.idx2card(i)

        entry = dict(card=card, gt_trick=v0)

        if avg_tricks is not None:
            entry["avg"] = avg_tricks[i]
            entry["min"] = min_tricks[i]
            entry["argmin"] = argmin_tricks[i]
            entry["max"] = max_tricks[i]
            entry["argmax"] = argmax_tricks[i]
            entry["mindist"] = min_dists[i]
            entry["argmin_dist"] = argmin_dists[i]

        tbl.append(entry)

    df = pd.DataFrame(tbl)
    print(df[headers])
    # print(tabulate(df, headers="keys", tablefmt='psql'))

    if avg_tricks is not None:
        err = np.linalg.norm(future_tricks_gt - avg_tricks)
        print("trick_l2_diff: ", err)
