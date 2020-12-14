import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
from collections import namedtuple

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = [64, 64]
ACTION_MIN = -1.0
ACTION_MAX = 1.0

A2COut = namedtuple("A2COut", ["actions", "log_probs", "entropy", "v"])


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs):
        return obs * torch.tanh(F.softplus(obs))


class A2CGaussian(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=HIDDEN_DIM, seed=42, activation=Mish
    ):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.activation = activation
        self.actor = nn.Sequential(
            *self._get_layers(state_dim, hidden_dim),
            nn.Linear(hidden_dim[-1], action_dim),
        )
        self.critic = nn.Sequential(
            *self._get_layers(state_dim, hidden_dim),
            nn.Linear(hidden_dim[-1], 1),
        )
        self.stds = nn.Parameter(torch.zeros(action_dim))
        self.to(DEVICE)

    def _get_layers(self, state_dim, hidden_dim):
        layers = [nn.Linear(state_dim, hidden_dim[0]), self.activation()]
        for layer in self._get_hidden_dims(hidden_dim):
            layers.append(layer)
            layers.append(self.activation())
        return layers

    def _get_hidden_dims(self, dims):
        return [nn.Linear(i, o) for i, o in zip(dims[:-1], dims[1:])]

    def forward(self, states):
        """
        Forward pass through network, returns actions, log_probs, v and entropy.
        States are numpy arrays of shape (n_agents, state_dim)
        We use tanh because our control actions need to have values between -1.0, 1.0
        """
        means = torch.tanh(self.actor(states))
        v = self.critic(states)
        dist = torch.distributions.Normal(means, F.softplus(self.stds))
        actions = dist.sample()
        return A2COut(
            actions=torch.clamp(actions, ACTION_MIN, ACTION_MAX),
            log_probs=dist.log_prob(actions).sum(dim=1),
            entropy=dist.entropy().sum(dim=1),
            v=v,
        )

    def checkpoint(self, path="checkpoint.pth"):
        dirs, fname = os.path.split(path)
        if dirs != "":
            os.makedirs(dirs, exist_ok=True)
        torch.save(self.state_dict(), path)
