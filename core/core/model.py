import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layers = [nn.Linear(state_size, hidden_layers[0])]
        for i, o in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.layers.append(nn.Linear(i, o))
        self.out = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)
