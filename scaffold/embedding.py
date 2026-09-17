import torch


class Embedding(torch.nn.Module):
    def __init__(self, count: int, channels: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(count, channels)

    def forward(self, index):
        return self.embedding(index)
