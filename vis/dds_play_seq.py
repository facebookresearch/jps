from redeal import *
from ctypes import Structure, c_int

class PlayTraceBin(Structure):
    """The playTraceBin struct."""

    _fields_ = [("number", c_int),
                ("suit", c_int * 52),
                ("rank", c_int * 52)]

def get_seq(s, trump, declarer):
    deal = [  [ [ global_defs.Rank[c] for c in holding ] for holding in hand.split(".") ] for hand in s[2:].split(" ") ]
    c_deal = dds.Deal.from_deal(deal, global_defs.Strain[trump], global_defs.Seat[declarer])

    # Call function
    _check_dll("AnalysePlayBin")

    trace_bin = PlayerTraceBin()
    trace_bin.number = 0
    dll.AnalyzePlayBin(c_deal, 


