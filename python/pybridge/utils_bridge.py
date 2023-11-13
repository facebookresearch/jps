import os
import sys

class ActionParser:
    def __init__(self):
        actions = [str(level) + strain for level in range(1, 8) for strain in ['C', 'D', 'H', 'S', 'N']]
        actions.append("P")
        actions.append("X")
        actions.append("XX")
        self.action2str = actions

    def print_distri(self, prob):
        s = ''
        for i in range(38):
            s =  s + self.action2str[i] + ":%.4f " % prob[i]
            if i % 5 == 4 or i == 37:
                print(s)
                s = ''


