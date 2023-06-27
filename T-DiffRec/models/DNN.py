import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


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
        steps=5,
        dropout=0.5,
        attention_weighting=False,
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
        self.param = torch.nn.Parameter(torch.rand(steps, emb_size))

        # NEW
        print(f"Embedding size: {self.time_emb_dim}")
        self.param_storage = []
        self.attention_weighting = attention_weighting

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
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

        if self.attention_weighting:
            d_hidden = 50  # TODO
            self.attention_w_0 = nn.Linear(d_hidden, 1, bias=False)
            self.attention_w_1 = nn.Linear(64, d_hidden, bias=False)
            self.attention_w_2 = nn.Linear(64, d_hidden, bias=False)
            self.attention_w_3 = nn.Linear(64, d_hidden, bias=False)
            # self.attention_b_a = nn.Parameter(d_hidden, 1) # TODO later

        self.drop = nn.Dropout(dropout)
        self.init_weights()

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

    def attention_net(
        self,
        all_user_interaction_embeddings,
        last_user_interaction_embedding,
        mean_user_interacted_embeddings,
    ):
        """
        Attention network to calculate the attention weights for each item after STAMP paper.
        Parameters
        :param all_user_interaction_embeddings: the embeddings of the items that the user has interacted with, shape: torch.Size([400, 64])
        :param last_user_interaction_embedding: the embedding of the last item that the user has interacted with, shape: torch.Size([400, 64])
        :param mean_user_interacted_embeddings: the embedding of the current item that the user is interacting with, shape: torch.Size([400, 1, 64])

        :return: the attention weights for each item, shape: # TODO
        """

        weighted_interactions = self.attention_w_1(all_user_interaction_embeddings)
        weighted_last_interaction = self.attention_w_2(last_user_interaction_embedding)
        weighted_mean_interaction = self.attention_w_3(mean_user_interacted_embeddings)

        combined_interactions = (
            weighted_interactions
            + weighted_last_interaction
            + weighted_mean_interaction
            # + self.attention_b_a # TODO add this
        )

        attentions = torch.nn.functional.softmax(combined_interactions, dim=1)
        attention_weights = self.attention_w_0(attentions)
        print(f"Attention weights shape: {attention_weights.shape}")

        return attention_weights

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        if self.attention_weighting:
            # a = self.attention_net(inputs)  # TODO
            # return h, a
            return h
        else:
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
