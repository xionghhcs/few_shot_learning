import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, embedding, max_length=128, embed_dim=200, hidden_size=100):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding))

        # conv
        self.conv = nn.Conv1d(embed_dim, hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)

    def forward(self, inputs): 
        # embedding
        text_embed = self.embedding(inputs)

        # encoder 
        text_encoded = self.conv(text_embed.transpose(1, 2))
        text_rep = self.pool(text_encoded)
        return text_rep.squeeze(2)
