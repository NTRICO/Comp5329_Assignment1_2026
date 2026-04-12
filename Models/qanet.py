import copy

import torch
import torch.nn as nn

from .conv import DepthwiseSeparableConv
from .embedding import Embedding
from .encoder import EncoderBlock
from .attention import CQAttention
from .heads import Pointer

class WordEmbedding(nn.Module):
    def __init__(self, word_mat):
        super().__init__()
        weight = torch.tensor(word_mat, dtype=torch.float32)

        self.register_buffer("null_emb", weight[0:1])
        self.oov_emb = nn.Embedding(1, weight.size(1))
        self.oov_emb.weight.data.copy_(weight[1:2])

        self.fixed_emb = nn.Embedding.from_pretrained(weight[2:], freeze=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fixed_emb.weight.new_zeros(*x.shape, self.fixed_emb.embedding_dim)

        null_mask = x == 0
        oov_mask = x == 1
        fixed_mask = x >= 2

        if null_mask.any():
            out[null_mask] = self.null_emb.expand(null_mask.sum(), -1)

        if oov_mask.any():
            out[oov_mask] = self.oov_emb.weight.expand(oov_mask.sum(), -1)

        if fixed_mask.any():
            out[fixed_mask] = self.fixed_emb(x[fixed_mask] - 2)

        return out


class QANet(nn.Module):
    """
    Args-driven QANet. No config.py dependency.
    Required args fields:
      - d_model, num_heads, glove_dim, char_dim, dropout, dropout_char
      - para_limit, ques_limit
      - pretrained_char (optional bool, default False)
    """

    def __init__(self, word_mat, char_mat, args):
        super().__init__()
        d_model = int(args.d_model)
        num_heads = int(args.num_heads)
        d_word = int(args.glove_dim)
        d_char = int(args.char_dim)
        dropout = float(args.dropout)
        dropout_char = float(args.dropout_char)
        len_c = int(args.para_limit)
        len_q = int(args.ques_limit)
        pretrained_char = bool(getattr(args, "pretrained_char", False))
        init_name = str(getattr(args, "init_name", "kaiming"))
        act_name = str(getattr(args, "activation", "relu"))
        norm_name = str(getattr(args, "norm_name", "layer_norm"))
        norm_groups = int(getattr(args, "norm_groups", 8))

        self.char_emb = nn.Embedding.from_pretrained(
            torch.tensor(char_mat, dtype=torch.float32), freeze=pretrained_char
        )
        self.word_emb = WordEmbedding(word_mat)

        self.emb = Embedding(
            d_word,
            d_char,
            dropout,
            dropout_char,
            init_name=init_name,
            act_name=act_name,
        )
        self.context_conv = DepthwiseSeparableConv(
            d_word + d_char, d_model, 5, init_name=init_name
        )
        self.question_conv = self.context_conv

        self.c_emb_enc = EncoderBlock(
            d_model,
            num_heads,
            dropout,
            conv_num=4,
            k=7,
            length=max(len_c, len_q),
            init_name=init_name,
            act_name=act_name,
            norm_name=norm_name,
            norm_groups=norm_groups,
        )
        self.q_emb_enc = self.c_emb_enc

        self.cq_att = CQAttention(d_model, dropout)
        self.cq_resizer = DepthwiseSeparableConv(
            d_model * 4, d_model, 1, init_name=init_name
        )

        base_enc = EncoderBlock(
            d_model,
            num_heads,
            dropout,
            conv_num=2,
            k=5,
            length=len_c,
            init_name=init_name,
            act_name=act_name,
            norm_name=norm_name,
            norm_groups=norm_groups,
        )
        self.model_enc_blks = nn.ModuleList([copy.deepcopy(base_enc) for _ in range(7)])

        self.out = Pointer(d_model)

    def forward(self, Cwid, Ccid, Qwid, Qcid):
        cmask = Cwid == 0  # True means PAD
        qmask = Qwid == 0

        Cw, Cc = self.word_emb(Cwid), self.char_emb(Ccid)
        Qw, Qc = self.word_emb(Qwid), self.char_emb(Qcid)

        C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        C = self.context_conv(C)
        Q = self.question_conv(Q)

        Ce = self.c_emb_enc(C, cmask)
        Qe = self.q_emb_enc(Q, qmask)

        X = self.cq_att(Ce, Qe, cmask, qmask)

        M1 = self.cq_resizer(X)
        for enc in self.model_enc_blks:
            M1 = enc(M1, cmask)

        M2 = M1
        for enc in self.model_enc_blks:
            M2 = enc(M2, cmask)

        M3 = M2
        for enc in self.model_enc_blks:
            M3 = enc(M3, cmask)

        p1, p2 = self.out(M1, M2, M3, cmask)
        return p1, p2
