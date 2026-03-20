import torch
from model import Transformer
from dataset import make_src_mask, make_tgt_mask, BOS, EOS

model = Transformer(20, 20)
model.eval()

src = torch.tensor([[1, 4, 5, 6, 2]])
src_mask = make_src_mask(src)

ys = torch.tensor([[BOS]])

for _ in range(10):
    tgt_mask = make_tgt_mask(ys)
    out = model(src, ys, src_mask, tgt_mask)
    next_word = out[:, -1].argmax(-1)

    ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

    if next_word.item() == EOS:
        break

print("prediction:", ys)
