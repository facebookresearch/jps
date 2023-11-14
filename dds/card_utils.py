import os
import random
import re
import sys

import numpy as np

IDX2RANK = "AKQJT98765432"
IMPORTANT_RANKS = "AKQJ"
RANK2IDX = {
    "A": 0,
    "K": 1,
    "Q": 2,
    "J": 3,
    "T": 4,
    "9": 5,
    "8": 6,
    "7": 7,
    "6": 8,
    "5": 9,
    "4": 10,
    "3": 11,
    "2": 12,
}

IDX2SUIT = "CDHS"
SUIT2IDX = {"C": 0, "D": 1, "H": 2, "S": 3}

IDX2STRAIN = "CDHSN"
STRAIN2IDX = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}

IDX2SEATS = "NESW"
SEATS2IDX = {"N": 0, "E": 1, "S": 2, "W": 3}

NUM_CARDS = 52
NUM_PLAYERS = 4
NUM_CARD_PER_PLAYER = 13
NUM_SUITS = 4
NUM_CARD_PER_SUIT = 13
NUM_STRAINS = 5
NUM_TRICKS = 13

SUIT2SYMBOL = {
    "C": u"\u2663",
    #"D": u"\u2666",
    "D": u"\u2662",
    "S": u"\u2660",
    #"H": u"\u2665",
    "H": u"\u2661",
    "N": "N"
}
NUM_BID_STATES = 39

def suit(card_idx):
    return IDX2SUIT[card_idx // NUM_CARD_PER_SUIT]


def rank(card_idx):
    return IDX2RANK[card_idx % NUM_CARD_PER_SUIT]

def idx2card(card_idx):
    return suit(card_idx) + rank(card_idx)


def card2idx(suit, rank):
    if isinstance(suit, str):
        suit = SUIT2IDX[suit]
    if isinstance(rank, str):
        rank = RANK2IDX[rank]
    return suit * NUM_CARD_PER_SUIT + rank


def dds_bit_rank(card_idx):
    # dds convention: A = 1 << 14, K = 1 << 13, etc.
    return 1 << (14 - card_idx % NUM_CARD_PER_SUIT)


def dds_rank2rank(dds_rank):
    # dds convention: A = 14, K = 13, etc
    return 14 - dds_rank


def suit_swap(suit):
    # dds convention is different from us (they use S=0, H=1, D=2, C=3)
    return (suit ^ 3)


def strain_swap(strain):
    # dds convention is different from us (they use S=0, H=1, D=2, C=3, N=4),
    # while we use CDHSN
    return (strain ^ 3) if strain < 4 else strain

def rank2idx(rank):
    return RANK2IDX[rank]

def suit2symbol(suit):
    assert suit in SUIT2SYMBOL
    return SUIT2SYMBOL[suit]

# class CardUtils:
#     def __init__(self):
#         self.rank2idx = dict(A=0, K=1, Q=2, J=3, T=4)
#         for i in range(2, 10):
#             self.rank2idx[str(i)] = 14 - i
#
#         self.idx2rank = "AKQJT98765432"
#         self.important_ranks = "AKQJ"
#
#         self.suit2idx = dict(C=0, D=1, H=2, S=3)
#         self.idx2suit = "CDHS"
#
#         self.idx2strain = "CDHSN"
#         self.strain2idx = {k: i for i, k in enumerate(self.idx2strain)}
#
#         self.seats = "NESW"
#         self.seat2idx = {k: i for i, k in enumerate(self.seats)}
#
#         self.num_players = 4
#         self.num_suits = 4
#         self.num_strains = 5
#         self.num_card_per_suit = 13
#         self.num_card_per_player = 13
#         self.num_cards = 52
#         self.num_tricks = 13
#
#     def card2idx(self, suit, idx):
#         if isinstance(suit, str):
#             suit = self.suit2idx[suit]
#
#         if isinstance(idx, str):
#             idx = self.rank2idx[idx]
#
#         return suit * self.num_card_per_suit + idx
#
#     def idx2card(self, card_idx):
#         return self.suit(card_idx) + self.rank(card_idx)
#
#     def rank(self, card_idx):
#         return self.idx2rank[card_idx % self.num_card_per_suit]
#
#     def dds_bit_rank(self, card_idx):
#         # dds convention: A = 1 << 14, K = 1 << 13, etc.
#         return 1 << (14 - card_idx % 13)
#
#     def dds_rank2rank(self, dds_rank):
#         # dds convention: A = 14, K = 13, etc
#         return 14 - dds_rank
#
#     def suit(self, card_idx):
#         return self.idx2suit[card_idx // self.num_card_per_suit]
#
#     def suitswap(self, suit_idx):
#         # dds convention is different from us (they use S=0, H=1, D=2, C=3)
#         return 3 - suit_idx
#
#     def strainswap(self, strain_idx):
#         # dds convention is different from us (they use S=0, H=1, D=2, C=3, N=4), while we use CDHSN
#         return 3 - strain_idx if strain_idx < 4 else strain_idx


vulMap = {"None": 0, "NS": 1, "EW": 2, "Both": 3}


def bidIdx2str(idx):
    if idx >= 35:
        return {35: "P ", 36: "X ", 37: "XX", 38: "-"}[idx]
    level = idx // 5 + 1
    strain = idx % 5
    return str(level) + "CDHSN" [strain]


def bidStr2idx(s):
    if s[0] == '(':
        s = s[1:-1]
    try:
        level = int(s[0])
        strain = dict(C=0, D=1, H=2, S=3, N=4)[s[1]]
        return (level - 1) * 5 + strain
    except:
        return dict(P=35, X=36, XX=37)[s.strip()]


def get_contract_declarer(dealer, seq):
    last_strain = '-'
    last_strain_player = -1
    first_call = [dict(), dict()]

    player = dealer
    for bid in seq:
        if bid[0] == '(':
            bid = bid[1:-1]

        curr_first_call = first_call[player % 2]

        strain = bid[1] if len(bid) > 1 else '-'

        if strain in "CDHSN":
            if strain not in curr_first_call:
                curr_first_call[strain] = player

            last_strain = str(strain)
            last_strain_player = player

        player = (player + 1) % 4

    if last_strain == '-':
        # No contract was made.
        declarer = None
    else:
        declarer = first_call[last_strain_player % 2][last_strain]

    return last_strain, declarer


# some functions to probe data.
def pbn2list(pbn):
    ''' Accept the following three types:

          [Deal "N:.63.AKQ987.A9732 A8654.KQ5.T.QJT6 J973.J98742.3.K4 KQT2.AT.J6542.85"]
          N:.63.AKQ987.A9732 A8654.KQ5.T.QJT6 J973.J98742.3.K4 KQT2.AT.J6542.85
          .63.AKQ987.A9732 A8654.KQ5.T.QJT6 J973.J98742.3.K4 KQT2.AT.J6542.85

        Output:
          Always starts with the order NESW
          [ ["", "63", "AKQ987", "A9732"], ["A8654", "KQ5", "T", "QJT6"], ["J973", "J98742", "3", "K4"], ["KQT2", "AT", "J6542", "85"] ]
    '''
    if pbn.startswith("[Deal"):
        m = re.match(r"^\[Deal\s\"(.*)\"", pbn)
        hands = m.group(1)
    else:
        hands = pbn

    start = 'N'
    items = hands.split(":")
    if len(items) > 1:
        start = items[0]
        hands = items[1]

    ret = [hand.split(".") for hand in hands.split(" ")]

    offset = SEATS2IDX[start]

    if offset == 0:
        return ret
    else:
        return ret[-offset:] + ret[:-offset]

def rotate_nfirst(pbn):
    seat = SEATS2IDX[pbn[0]]
    hands = pbn[2:].split(" ")
    hands = hands[-seat:] + hands[:-seat]
    return IDX2SEATS[0] + ":" + " ".join(hands)


def get_strength(deal):
    tbl = {'A': 4, 'K': 3, 'Q': 2, 'J': 1}

    def _strength(suit):
        return sum(tbl.get(c, 0) for c in suit)

    return [[_strength(suit) for suit in hand] for hand in deal]


def suit_transfer(s1, s2):
    ''' s1 -> s2 '''
    idx = random.randint(0, len(s1) - 1)
    s2.append(s1[idx])
    del s1[idx]


def compare(d1, d2):
    for dd1, dd2 in zip(d1, d2):
        for a1, a2 in zip(dd1, dd2):
            if a1 != a2:
                return False
    return True

def make_random_card_map():
    card_map = [None] * NUM_CARDS
    p = list(range(NUM_CARDS))
    random.shuffle(p)
    for i in range(NUM_PLAYERS):
        for j in range(NUM_CARD_PER_PLAYER):
            card_map[p[i * NUM_CARD_PER_PLAYER + j]] = i
    return card_map


def make_card_map(state):
    card_map = [None] * NUM_CARDS
    if isinstance(state, str):
        state = pbn2list(state)
    for i, p in enumerate(state):
        for j, s in enumerate(p):
            for c in s:
                idx = card2idx(suit_swap(j), c)
                card_map[idx] = i
    return card_map


def pbn_dist(pbn1, pbn2):
    m1 = make_card_map(pbn1)
    m2 = make_card_map(pbn2)
    return np.sum(np.not_equal(m1, m2))
