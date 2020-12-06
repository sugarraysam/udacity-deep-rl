import os
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZES = [64, 64]


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=HIDDEN_SIZES, seed=42):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            *self._get_hidden_layers(hidden_sizes),
        )
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def _get_hidden_layers(self, sizes):
        return [nn.Linear(i, o) for i, o in zip(sizes[:-1], sizes[1:])]

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

    def checkpoint(self, path="checkpoint.pth"):
        dirs, fname = os.path.split(path)
        if dirs != "":
            os.makedirs(dirs, exist_ok=True)
        torch.save(self.state_dict(), path)
