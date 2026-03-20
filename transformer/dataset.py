import torch

PAD = 0
BOS = 1
EOS = 2

def make_src_mask(src):
    return (src != PAD).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt):
    batch, seq = tgt.size()
    pad_mask = (tgt != PAD).unsqueeze(1).unsqueeze(2)
    nopeak = torch.tril(torch.ones(seq, seq)).bool()
    return pad_mask & nopeak
