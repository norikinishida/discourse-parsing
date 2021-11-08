from collections import Iterable

import torch
import torch.nn as nn
import torch.nn.init as init


def make_embedding(dict_size, dim, std=0.02):
    """
    Parameters
    ----------
    dict_size: int
    dim: int
    std: float

    Returns
    -------
    torch.nn.modules.sparse.Embedding
    """
    emb = nn.Embedding(dict_size, dim)
    init.normal_(emb.weight, std=std)
    return emb


def make_linear(input_dim, output_dim, bias=True, std=0.02):
    """
    Parameters
    ----------
    input_dim: int
    output_dim: int
    bias: bool
    std: float

    Returns
    -------
    torch.nn.modules.linear.Linear
    """
    linear = nn.Linear(input_dim, output_dim, bias)
    init.normal_(linear.weight, std=std)
    if bias:
        init.zeros_(linear.bias)
    return linear


def make_mlp(input_dim, hidden_dims, output_dim, dropout):
    """
    Parameters
    ----------
    input_dim: int
    hidden_dims: list[int] or int or None
    output_dim: int
    dropout: nn.Dropout

    Returns
    -------
    torch.nn.modules.container.Sequential
    """
    if (hidden_dims is None) or (hidden_dims == 0) or (hidden_dims == []) or (hidden_dims == [0]):
        return make_linear(input_dim, output_dim)

    if not isinstance(hidden_dims, Iterable):
        hidden_dims = [hidden_dims]

    mlp = [make_linear(input_dim, hidden_dims[0]), nn.ReLU(), dropout]
    for i in range(1, len(hidden_dims)):
        mlp += [make_linear(hidden_dims[i-1], hidden_dims[i]), nn.ReLU(), dropout]
    mlp.append(make_linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*mlp)


def make_mlp_hidden(input_dim, hidden_dim, dropout):
    """
    Parameters
    ----------
    input_dim: int
    hidden_dim: int
    dropout: nn.Dropout

    Returns
    -------
    torch.nn.modules.container.Sequential
    """
    mlp = [make_linear(input_dim, hidden_dim), nn.ReLU(), dropout]
    return nn.Sequential(*mlp)


class Biaffine(nn.Module):

    def __init__(self, input_dim, output_dim=1, bias_x=True, bias_y=True):
        """
        Parameters
        ----------
        input_dim: int
        output_dim: int
        bias_x: bool
        bias_y: bool
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim+bias_x, input_dim+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"input_dim={self.input_dim}, output_dim={self.output_dim}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        # nn.init.zeros_(self.weight)
        init.normal_(self.weight, std=0.02)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x: torch.Tensor(shape=(batch_size, seq_len, input_dim))
        y: torch.Tensor(shape=(batch_size, seq_len, input_dim))

        Returns
        -------
        torch.Tensor(shape=(batch_size, output_dim, seq_len, seq_len))
            A scoring tensor of shape ``[batch_size, output_dim, seq_len, seq_len]``.
            If ``output_dim=1``, the dimension for ``output_dim`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1) # (batch_size, seq_len, input_dim+1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1) # (batch_size, seq_len, input_dim+1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y) # (batch_size, output_dim, seq_len, seq_len)

        return s


