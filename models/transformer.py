import torch.nn as nn

from .attention import MultiHeadAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadAttention(
            embed_dim,
            num_heads,
            dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP(
            embed_dim,
            embed_dim * mlp_ratio,
            dropout
        )

    def forward(self, x, mask=None):
        x = x + self.attention(
            self.norm1(x),
            mask
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x
