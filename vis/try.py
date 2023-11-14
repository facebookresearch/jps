import tkinter as tk
from tkinter import messagebox as tkm
from tkinter import ttk
from tkinter import font
# from cairosvg import svg2png

from PIL import Image, ImageTk, PngImagePlugin

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import itertools
import json
import re
import io

import sys

sys.path.append("../python")
sys.path.append("../dds")
from prob_data import Client
import importlib
from sample_utils import compute_dd_table_backup, DealWalk
import card_utils

def bid2symbol(bid):
    if bid.startswith('('):
        bid = bid[1:-1]

    contract_trick = None
    if bid[-1] not in ('P', 'X'):
        contract_trick = int(bid[0])
        bid = bid[:-1] + card_utils.suit2symbol(bid[-1])

    return bid, contract_trick


def print_contract_table(ddt):
    assert ddt.size == 20
    if ddt.shape[0] == 4:
        ddt = ddt.reshape(20)
    count = 0
    c_list = "   " + "   ".join([ " " + card_utils.suit2symbol(c) for c in card_utils.IDX2STRAIN]) + "\n"
    for p in card_utils.IDX2SEATS:
        contracts = []
        for s in card_utils.IDX2STRAIN:
            trick = str(int(ddt[count]))
            count += 1
            if len(trick) == 1:
                trick = " " + trick
            contracts.append(trick)

        c_list += p + ": " + "   ".join(contracts) + "\n"
    return c_list

def load_suit_imgs(filename, sz):
    img = Image.open(filename)
    w, h = img.size

    spade = img.crop([0, 0, w / 2, h / 2])
    heart = img.crop([w / 2, 0, w, h / 2])
    diamond = img.crop([0, h / 2, w / 2, h])
    club = img.crop([w / 2, h / 2, w, h])

    suits = [ spade, heart, diamond, club ]
    return [ ImageTk.PhotoImage(suit.resize((sz, sz), Image.ANTIALIAS)) for suit in suits ]

class Player:
    def __init__(self, canvas, name, x0, y0):
        self.canvas = canvas
        self.name = name

        display_font = font.Font(family='Helvetica', size=30)
        caption_font = font.Font(family='Helvetica', size=30, weight="bold")

        self.margin = 5
        self.sz = 30

        self.suits = load_suit_imgs("./deck/playing-card-suits.jpg", self.sz)
        self.suits2 = load_suit_imgs("./deck/playing-card-suits2.jpg", self.sz)

        self.suit_ids = []
        self.suit_imgs = []
        self.suit_selected = [False] * len(self.suits)

        self.caption_id = canvas.create_text(x0, y0, anchor=tk.NW, text=name, font=caption_font)
        self.canvas.tag_bind(self.caption_id, '<Double-1>', lambda e: self.onCaptionClick())

        for i, suit in enumerate(self.suits):
            y = y0 + (i + 1) * (self.sz + self.margin)
            img_obj = canvas.create_image(x0, y, anchor=tk.NW, image=suit)
            canvas.tag_bind(img_obj, '<Double-1>', lambda e, i=i: self.onSuitClick(i))
            text_id = canvas.create_text(x0 + self.sz + self.margin, y, anchor=tk.NW, font=display_font)
            self.suit_ids.append(text_id)
            self.suit_imgs.append(img_obj)

    def draw(self, caption, hand):
        # Hand is in the form of e.g., 92.KT3.Q932.8642
        # hand might have suffix, e.g., n:
        self.canvas.itemconfigure(self.caption_id, text=self.name + "  (" + caption + ")")

        for suit, text_id in zip(hand.split("."), self.suit_ids):
            if suit == "":
                suit = "---"
            self.canvas.itemconfigure(text_id, text=suit)

    def onSuitClick(self, i):
        # Toggle boundary.
        self.suit_selected[i] = not self.suit_selected[i]
        img = self.suits2[i] if self.suit_selected[i] else self.suits[i]
        self.canvas.itemconfigure(self.suit_imgs[i], image=img)

    def onCaptionClick(self):
        # revert all.
        for i in range(4):
            self.onSuitClick(i)

class Bidding:
    def __init__(self, canvas, orders, x0, y0):
        # 6 x 4 text matrix.
        margin_x = 70
        margin_y = 30
        self.bidding_font_caption = font.Font(family="Monaco", size=22, weight="bold")
        self.bidding_font = font.Font(family="Monaco", size=20)
        self.bidding_font_small = font.Font(family="Monaco", size=16)

        self.round = 7

        self.bidding_labels = [ [ None for i in range(len(card_utils.IDX2SEATS)) ] for j in range(self.round) ]
        self.bidding_infos = [ [ dict(index=-1, s=tk.StringVar()) for i in range(len(card_utils.IDX2SEATS)) ] for j in range(self.round) ]

        for j in range(self.round + 1):
            y = y0 + margin_y * j
            for i, o in enumerate(card_utils.IDX2SEATS):
                x = x0 + margin_x * i
                if j == 0:
                    canvas.create_text(x, y, text=o, anchor=tk.NW, font=self.bidding_font_caption)
                else:
                    s = self.bidding_infos[j - 1][i]["s"]

                    label = tk.Label(canvas, textvariable=s, anchor=tk.NW, font=self.bidding_font)
                    label.place(x=x, y=y)
                    label.bind("<Enter>", lambda event, j=j, i=i: self.on_enter(event, j - 1, i))
                    label.bind("<Leave>", self.on_leave)
                    self.bidding_labels[j - 1][i] = label

        self.summary_id = canvas.create_text(x0, y0 + margin_y * (self.round + 1), text="", anchor=tk.NW, font=self.bidding_font)
        self.explanation_id = canvas.create_text(x0, y0 + margin_y * (self.round + 2), text="", anchor=tk.NW, font=self.bidding_font_small)
        self.canvas = canvas

    def on_enter(self, event, row, col):
        k = self.bidding_infos[row][col]["index"]
        if k < 0:
            return

        event.widget.config(bg="green")
        if self.other_choices is None:
            return

        choices = []
        for c in self.other_choices[k]:
            bid, _ = bid2symbol(c["bid"])
            if c["prob"] > 0.01:
                choices.append(f'{bid}: {c["prob"]:.2f}')

        text = "\n".join(choices)
        self.canvas.itemconfigure(self.explanation_id, text=text)

    def on_leave(self, event):
        event.widget.config(bg="white")
        self.canvas.itemconfigure(self.explanation_id, text="")

    def set_seq(self, dealer, seq):
        final_contract = 0
        row = 0

        for a in self.bidding_infos:
            for b in a:
                b["index"] = -1
                b["s"].set("    ")

        for k, bid in enumerate(seq):
            bid, contract = bid2symbol(bid)
            if contract is not None:
                final_contract = contract

            # Set it to bidding_ids
            self.bidding_infos[row][dealer]["index"] = k
            self.bidding_infos[row][dealer]["s"].set(f"{bid:<3}")

            dealer += 1
            if dealer == 4:
                dealer = 0
                row += 1

        self.seq = seq
        self.final_contract_trick = final_contract + 6

    def set_others(self, this_bid):
        if "trickTaken" in this_bid:
            trick_taken = this_bid["trickTaken"]
            over_trick = trick_taken - self.final_contract_trick
            self.canvas.itemconfigure(self.summary_id, text=f"Trick Taken: {trick_taken} ({over_trick})")

        if "otherSeq" in this_bid:
            other_choices = this_bid["otherSeq"]
            assert len(self.seq) == len(other_choices), f"{len(self.seq)} != {len(other_choices)}"
            self.other_choices = other_choices
        else:
            self.other_choices = None

class App:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(master, width = 1000, height = 600)
        self.canvas.pack()

        locs = dict(N=(220, 0), E=(420, 200), S=(220, 400), W=(20, 200))
        self.players = { k : Player(self.canvas, k, *v) for k, v in locs.items() }

        self.bidding = Bidding(self.canvas, card_utils.IDX2STRAIN, 700, 0)

        self.button_font = font.Font(family="Monaco", size=20)
        self.text_font = font.Font(size=16)
        self.ddt_font = font.Font(family="Monaco", size=16)

        self.pw = tk.PanedWindow(master)

        self.next_bid = tk.Button(self.pw, text="Next Bid", command=self.onNextBid, font=self.button_font)
        self.next_bid.pack(side=tk.LEFT)
        self.prev_sample = tk.Button(self.pw, text="Prev Sample", command=lambda: self.onNextSample(-1), font=self.button_font)
        self.prev_sample.pack(side=tk.LEFT)
        self.next_sample = tk.Button(self.pw, text="Next Sample", command=lambda: self.onNextSample(1), font=self.button_font)
        self.next_sample.pack(side=tk.LEFT)
        self.resample = tk.Button(self.pw, text="Resample", command=self.onResample, font=self.button_font)
        self.resample.pack(side=tk.LEFT)
        self.reset_curr_sample = tk.Button(self.pw, text="Reset Sample", command=lambda: self.onResample(True), font=self.button_font)
        self.reset_curr_sample.pack(side=tk.LEFT)

        self.pw.pack()

        #self.cmd_text = tk.StringVar()
        #self.cmd_text.set("bid_seq 1S-P")
        #self.cmd_entry = tk.Text(master, textvariable=self.cmd_text, font=self.button_font)
        self.cmd_entry = tk.Text(master, font=self.button_font)
        self.cmd_entry.pack()

        self.cmd_button = tk.Button(master, text="SendCommand", command=self.onSendCmd, font=self.button_font)
        self.cmd_button.pack()

        self.ddt_id = self.canvas.create_text(700, 500, anchor=tk.NW, text="", font=self.ddt_font)
        self.overview_id = self.canvas.create_text(10, 10, anchor=tk.NW, text="", font=self.text_font)

        self.stats_id = self.canvas.create_text(10, 500, anchor=tk.NW, text="", font=self.text_font)

        self.fig_suit_len = plt.figure(figsize=(6,5), dpi=100)
        self.fig_suit_len_ax = self.fig_suit_len.add_subplot(111)
        self.suit_len = None

        self.client = Client()
        self.stats = None

    def set_stats(self):
        if self.stats is None:
            text = ""
        elif isinstance(self.stats, str):
            text = self.stats
        else:
            text = "\n".join([ f"{k}={v}" for k, v in self.stats.items() ])

        self.canvas.itemconfigure(self.stats_id, text=text)

        if self.suit_len is not None:
            try:
                self.suit_len.get_tk_widget().pack_forget()
            except AttributeError:
                pass

        # Compute the stats for dealer's distribution.
        '''
        all_hcps = [ s["hcp"][s["dealer"]] for s in self.states ]
        self.fig_suit_len_ax.clear()
        self.fig_suit_len_ax.hist(all_hcps, bins=50)
        self.fig_suit_len_ax.set_xlabel("HCP")
        self.fig_suit_len_ax.set_ylabel("Count")

        self.suit_len = FigureCanvasTkAgg(self.fig_suit_len, self.master)
        self.suit_len.get_tk_widget().pack()
        '''

    def onResample(self, reset=False):
        state = self.states[self.state_idx]

        if not reset:
            # Resample the deal (a few suits will be fixed) and rerun DDS
            w = state["state"]
            suits_lock = [ self.players[p].suit_selected for p in card_utils.IDX2SEATS ]
            mask = w.get_card_map_mask(suits_lock)
            w.resample_with_mask(mask, suits_in_lock=suits_lock)
        else:
            state["state"] = DealWalk.from_card_map(state["card_map"])

        del state["DDT"]
        self.set_deal()

    def set_deal(self):
        state = self.states[self.state_idx]
        hands = state["state"].get_pbn_noprefix()

        for order, hand, hcp in zip(card_utils.IDX2SEATS, hands.split(" "), state["state"].get_hcp()):
            self.players[order].draw(f"HCP: {hcp}", hand)

        # Basic information
        info = "Batch idx: %d/%d\nRecord idx: %d\nVul: %s" % (self.state_idx, len(self.states), state.get("idx", 0), state.get("vul", 'invalid'))
        self.canvas.itemconfigure(self.overview_id, text=info)

        # ddt results.
        if "DDT" not in state:
            # Compute it on the fly
            state["DDT"] = compute_dd_table_backup([state["state"]])[0]

        ddt_text = print_contract_table(state["DDT"])
        self.canvas.itemconfigure(self.ddt_id, text=ddt_text)

    def onNextSample(self, delta):
        self.state_idx += len(self.states) + delta
        self.state_idx = self.state_idx % len(self.states)

        self.set_deal()
        self.bid_idx = -1
        self.onNextBid()


    def onNextBid(self):
        state = self.states[self.state_idx]

        if "bidd" not in state:
            return

        self.bid_idx += 1
        self.bid_idx = self.bid_idx % len(state["bidd"])

        dealer = state["dealer"]
        this_bid = state["bidd"][self.bid_idx]

        seq = this_bid["seq"]
        self.bidding.set_seq(dealer, seq)
        self.bidding.set_others(this_bid)

    def onSendCmd(self):
        # s = r"""{"bidd":[{"rawNSScore":140,"seq":["1C","(P)","2H","(P)","P","(P)"],"trickTaken":9},{"rawNSScore":-100,"seq":["(1S)","P","(2C)","P","(2S)","P","(3D)","P","(3H)","P","(3N)","P","(P)","P"],"trickTaken":7}],"dealer":0,"par_score":140,"reward":0.25,"state":"[Deal \"N:AQT986.J2.J9.954 754.T75.875.KQJ7 .K9864.AKQT43.32 KJ32.AQ3.62.AT86\"]","state_display":"\nSeat ♠   ♥   ♦   ♣   HCP   Actual Hand\n0    6   2   2   3   8    ♠AQT986 ♥J2 ♦J9 ♣954 \n1    3   3   3   4   6    ♠754 ♥T75 ♦875 ♣KQJ7 \n2    0   5   6   2   12   ♠ ♥K9864 ♦AKQT43 ♣32 \n3    4   3   2   4   14   ♠KJ32 ♥AQ3 ♦62 ♣AT86 \n","vul":"EW"}"""
        #cmd_text = self.cmd_text.get()
        cmd_text = self.cmd_entry.get("1.0", "end")
        if cmd_text.startswith("local"):
            state = dict()
            # just parse it now.
            _, cmd, content = cmd_text.strip().split(" ", 2)
            if cmd == "tab":
                # Construct state from gt_tabs.
                content = eval(content.replace(" ",","))
                deal = DealWalk.from_card_map(content)
                state["card_map"] = content
                state["state"] = deal
                self.states = [ state ]
            elif cmd == "pbn":
                deal = DealWalk.from_pbn(content)
                state["card_map"] = content
                state["state"] = deal
                self.states = [ state ]

            elif cmd == "json":
                states = json.loads(content)
                for state in states:
                    state["state"] = DealWalk.from_card_map(state["card_map"])
                self.states = states
            else:
                raise NotImplementedError(f"cmd: {cmd} is not implemented in local")

            self.stats = None
        else:
            ret = self.client.query(cmd_text)
            # print(f"Received: f{s}")
            prob_module = importlib.import_module("prob_data")
            prober_cls = getattr(prob_module, ret['_clsname'])
            self.states, self.stats = prober_cls.convert(ret)
            self.set_stats()

        self.state_idx = -1
        self.onNextSample(1)


root = tk.Tk()
root.wm_title('Bridge Visualization')
# root.wm_attributes('-alpha', .5)
root.wm_attributes("-topmost", 1);
#img = tk.PhotoImage(file='clock.gif')
#root.tk.call('wm', 'iconphoto', root._w, img)

app = App(root)
root.mainloop()
