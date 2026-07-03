import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embed_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            max_length,
            embed_dim
        )

    def forward(self, token_embeddings):
        batch_size, seq_length, _ = token_embeddings.shape

        positions = torch.arange(
            seq_length,
            device=token_embeddings.device
        )

        positions = positions.unsqueeze(0).expand(batch_size, seq_length)

        position_embeddings = self.embedding(positions)

        return token_embeddings + position_embeddings