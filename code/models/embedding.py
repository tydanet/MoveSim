# coding: utf-8

import torch.nn as nn

class Embedding(nn.Module):
    """Common embedding network.
    """

    def __init__(self, total_locations, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=total_locations, embedding_dim=embedding_dim)

    def forward(self, x):
        return self.embedding(x)
