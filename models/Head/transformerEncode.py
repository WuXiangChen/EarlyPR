import torch
from torch import nn
from .TransModel.model.encoder import Encoder
from CONSTANT.TransModel_Constant import *

class transHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.encoder = Encoder(
                               d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers)

    def forward(self, src):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        return enc_src

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        return src_mask
