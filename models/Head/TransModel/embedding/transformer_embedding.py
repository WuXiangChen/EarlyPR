import torch
from torch import nn

from ..embedding.positional_encoding import PositionalEncoding
from ..embedding.token_embeddings import TokenEmbedding

class TransformerEmbedding(nn.Module):


    def __init__(self,vocab_size, d_model, max_len, drop_prob, device):

        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)


    def forward(self, x):
        tok_emb = x.unsqueeze(dim=-1).expand((x.shape[0], x.shape[1], self.d_model))
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
