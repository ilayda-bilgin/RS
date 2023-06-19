import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """generate positional encodings for a transformer model

        Args:
            d_model (int): _description_
            dropout (float, optional): _description_. Defaults to 0.1.
            max_len (int, optional): _description_. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        max_len_pos_enc: int = 5000,
    ):
        """Transformer model to learn a weighting of the input features.

        Args:
            ntoken (int): Number of tokens (vocab size noramlly).
            d_model (int): Number of expected features in the input (required).
            nhead (int): Number of heads in ``nn.MultiheadAttention``.
            d_hid (int): FF size in nn.TransformerEncoder.
            nlayers (int): Number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``.
            dropout (float, optional): Probability of dropout. Defaults to 0.5.
        """
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len_pos_enc)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.max_len_pos_enc = max_len_pos_enc

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size, ntoken]``
            src_mask: Tensor, shape ``[seq_len, seq_len]`` Defaults to None.

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        print(f"Input shape: {src.shape}")
        src_transformed = src.unsqueeze(-1)
        print(f"Input shape after unsqueezing: {src_transformed.shape}")

        # permute from [batch_size, features, n_token] to [features, batch_size, ntoken]
        src_transformed = src_transformed.permute(1, 0, 2)
        print(
            f"Input shape after permuting: {src_transformed.shape}"
        )  # [34395, 400, 1]

        # keep only the last max_len_pos_enc timesteps of the interactions
        src_transformed = src_transformed[-self.max_len_pos_enc :, :, :]
        print(
            f"Input shape after keeping only last max_len_pos_enc: {src_transformed.shape}"
        )

        src_transformed = self.pos_encoder(
            src_transformed
        )  # expands last dim to d_model
        output = self.transformer_encoder(
            src_transformed, src_mask
        )  # mask is None, because not needed here
        print(f"After transformer encoder: {output.shape}")
        output = self.linear(output)  # reduces the last dim again to ntoken=1.
        return output


class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        emb_size,
        time_type="cat",
        norm=False,
        dropout=0.5,
        transformer_weighting=False,
    ):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert (
            out_dims[0] == in_dims[-1]
        ), "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.transformer_weighting = transformer_weighting

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]):
                print(f"d_in: {d_in}, d_out: {d_out}")

        else:
            raise ValueError(
                "Unimplemented timestep embedding type %s" % self.time_type
            )
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])
            ]
        )
        self.out_layers = nn.ModuleList(
            [
                nn.Linear(d_in, d_out)
                for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])
            ]
        )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

        # NEW: Transformer encoder
        if self.transformer_weighting:
            self.transformer_encoder = TransformerModel(
                ntoken=1,
                d_model=2,
                nhead=2,
                d_hid=2,
                nlayers=2,
                dropout=0.5,
                max_len_pos_enc=5000,  # TODO make this dynamic
            )

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):  # HERE
        # TODO: decide what to do with the existing time embedding:
        # Is it compatible with the transformer? or do we use only the transformer's positional encoding?

        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)  # dropout

        # NEW: Transformer encoder
        if self.transformer_weighting:
            weigths = self.transformer_encoder(x)

            # reweight the input (using elementwise multiplication)
            x = x * weigths

        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
