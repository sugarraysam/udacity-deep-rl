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

HIDDEN_DIM = [32]
ACTION_MIN = -1.0
ACTION_MAX = 1.0

A2COut = namedtuple("A2COut", ["actions", "log_probs", "entropy", "v"])


class A2CGaussian(nn.Module):
    """
    All non-output layers are shared between actor and critic.
    Actor output is of shape (2, action_dim), which corresponds to a vector for means
    and a vector for standard deviations of an underlying normal distribution.
    Critic tries to approximate the state-value function.
    """

    def __init__(
        self, state_dim, action_dim, hidden_dim=HIDDEN_DIM, seed=42, activation=nn.ReLU
    ):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        self.base = nn.Sequential(*self._get_layers(state_dim, hidden_dim))
        self.actor_out = nn.Linear(hidden_dim[-1], action_dim * 2)
        self.critic_out = nn.Linear(hidden_dim[-1], 1)
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
        States are numpy arrays of shape (n_agents, state_dim).
        We use tanh on the means because our control actions need to have values between -1.0, 1.0.
        We use softplus on the sigmas, so they are greater than 0.
        """
        x = self.base(states)
        a_out = self.actor_out(x)
        c_out = self.critic_out(x)
        means, sigmas = a_out[:, : self.action_dim], a_out[:, self.action_dim :]
        dist = torch.distributions.Normal(torch.tanh(means), F.softplus(sigmas))
        actions = dist.sample()
        return A2COut(
            actions=torch.clamp(actions, ACTION_MIN, ACTION_MAX),
            log_probs=dist.log_prob(actions).sum(dim=1),
            entropy=dist.entropy().sum(dim=1),
            v=c_out,
        )

    def checkpoint(self, path="checkpoint.pth"):
        dirs, fname = os.path.split(path)
        if dirs != "":
            os.makedirs(dirs, exist_ok=True)
        torch.save(self.state_dict(), path)
