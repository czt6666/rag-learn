import torch
import torch.nn as nn
from model import Transformer
from dataset import make_src_mask, make_tgt_mask, PAD

# toy vocab
SRC_VOCAB = 20
TGT_VOCAB = 20

model = Transformer(SRC_VOCAB, TGT_VOCAB)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

# toy data
src = torch.tensor([[1, 4, 5, 6, 2]])
tgt = torch.tensor([[1, 7, 8, 9, 2]])

for epoch in range(200):
    optimizer.zero_grad()

    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]

    src_mask = make_src_mask(src)
    tgt_mask = make_tgt_mask(tgt_input)

    out = model(src, tgt_input, src_mask, tgt_mask)

    loss = criterion(
        out.view(-1, TGT_VOCAB),
        tgt_output.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")
