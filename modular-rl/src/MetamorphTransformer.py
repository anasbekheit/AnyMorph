import math

import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity="relu"):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if nonlinearity == "relu":
            layers.append(nn.ReLU())
        elif nonlinearity == "tanh":
            layers.append(nn.Tanh())

    if not final_nonlinearity:
        layers.pop()
    return nn.Sequential(*layers)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class PositionalEncoding1D(nn.Module):

    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


def init_pos_embedding(type, dmodel, seq_len):
    if type == "learnt":
        raise NotImplementedError("Unsupported yet.")
    elif type == "abs":
        return PositionalEncoding1D(dmodel, seq_len)
    return lambda x: x


class MetaMorphTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            decoder_out_dim,
            seq_len,
            d_model,
            nhead,
            nfeed_fwd,
            nlayers,
            dropout,
            initrange=0.1,
            pos_type='abs',

    ):
        super(MetaMorphTransformer, self).__init__()
        self.d_model = d_model
        self.limb_embed = nn.Linear(input_dim, d_model)

        self.pos_embedding = init_pos_embedding(pos_type, d_model, seq_len)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nfeed_fwd, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        decoder_input_dim = d_model

        self.decoder = make_mlp_default([decoder_input_dim, decoder_out_dim], final_nonlinearity=False)
        self.init_weights(initrange)

    def init_weights(self, initrange):
        self.limb_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder[-1].bias.data.zero_()
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, obs, obs_mask):
        # (batch_size, seq_length, limb_obs_size) -> (batch_size, seq_length, d_model)
        obs_embed = self.limb_embed(obs) * math.sqrt(self.d_model)
        batch_size, _, _ = obs_embed.shape
        obs_embed = obs_embed.permute(1, 0, 2)
        obs_embed = self.pos_embedding(obs_embed)
        obs_embed = self.transformer_encoder(obs_embed, src_key_padding_mask=obs_mask)
        obs_embed = self.decoder(obs_embed).permute(1, 0, 2)
        obs_embed = obs_embed.reshape(batch_size, -1)
        return obs_embed
