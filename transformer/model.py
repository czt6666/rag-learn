import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

def attention(q, k, v, mask=None):
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.heads = heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)

        q = self.w_q(q).view(batch, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch, -1, self.heads, self.d_k).transpose(1, 2)

        x = attention(q, k, v, mask)

        x = x.transpose(1, 2).contiguous().view(batch, -1, self.heads * self.d_k)
        return self.w_o(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # Self-Attention
        x = self.norm1(x + self.attn(x, x, x, mask))
        # Feed Forward
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads)
        self.enc_attn = MultiHeadAttention(d_model, heads)
        self.ff = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # Masked Self-Attention（防偷看未来）
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        # Encoder-Decoder Attention
        x = self.norm2(x + self.enc_attn(x, enc_out, enc_out, src_mask))
        # Feed Forward
        x = self.norm3(x + self.ff(x))
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, N=2, heads=4, d_ff=256):
        super().__init__()

        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model)

        self.encoders = nn.ModuleList(
            [EncoderLayer(d_model, heads, d_ff) for _ in range(N)]
        )
        self.decoders = nn.ModuleList(
            [DecoderLayer(d_model, heads, d_ff) for _ in range(N)]
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.pos(self.src_embed(src))
        tgt = self.pos(self.tgt_embed(tgt))

        for enc in self.encoders:
            src = enc(src, src_mask)

        for dec in self.decoders:
            tgt = dec(tgt, src, src_mask, tgt_mask)

        return self.fc_out(tgt)
