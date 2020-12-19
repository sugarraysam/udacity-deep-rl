from control.model import DEVICE
from collections import namedtuple

import torch
import numpy as np

# Hyperparameters
DISCOUNT = 0.99  # discount factor
TAU = 1.0  # for advantages (GAE)
ENTROPY_LOSS_WEIGHT = 0.01  # importance of entropy loss (beta)
CRITIC_LOSS_WEIGHT = 1.0  # importance of critic loss

Sample = namedtuple("sample", ["rewards", "dones", "log_probs", "v", "entropy"])


class MiniBatch:
    """
    Wrapper around required parameters for a batch.
    """

    def __init__(
        self,
        discount=DISCOUNT,
        tau=TAU,
        e_weight=ENTROPY_LOSS_WEIGHT,
        c_weight=CRITIC_LOSS_WEIGHT,
    ):
        self.samples = []
        # Learning hyper params
        self.discount = discount
        self.tau = tau
        self.e_weight = e_weight
        self.c_weight = c_weight
        self.rewards_normalizer = Normalizer()

    def append(self, sample):
        self.samples.append(sample)

    def compute_loss(self, v_next):
        """
        Compute loss using the accumulated samples. Reset stored samples.
        All tensors are of shape (batch_size, n_agents) except for v, because we appended v_next.
        """
        rewards, dones, log_probs, v, entropy = self._parse_samples(v_next)
        advantages, returns = self._compute_advantages_and_returns(rewards, dones, v)

        a_loss = -(log_probs * advantages).mean()
        c_loss = 0.5 * (returns - v[:-1]).pow(2).mean()
        e_loss = entropy.mean()
        return a_loss - self.e_weight * e_loss + self.c_weight * c_loss

    def _compute_advantages_and_returns(self, rewards, dones, v):
        t, n_agents = rewards.shape[0], rewards.shape[1]
        all_advantages, all_returns = [], []

        advantages = torch.zeros(n_agents)
        returns = v[-1].clone()
        for i in reversed(range(t)):
            returns = rewards[i] + self.discount * dones[i] * returns
            td_error = rewards[i] + self.discount * dones[i] * v[i + 1] - v[i]
            advantages = advantages * self.tau * self.discount * dones[i] + td_error

            all_advantages.append(advantages.unsqueeze(0).clone())
            all_returns.append(returns.unsqueeze(0).clone())

        # reverse and return as tensors
        return torch.cat(all_advantages[::-1]), torch.cat(all_returns[::-1])

    def _parse_samples(self, v_next):
        """
        Prepare parameters as torch tensors. Rewards and dones are plain python lists.
        Dones are 0 if true, and 1 if false, so we can use them in equations.
        """
        rewards, dones, log_probs, v, entropy = zip(*self.samples)
        self.samples = []
        t, n_agents = len(rewards), len(rewards[0])

        rewards = self.rewards_normalizer(rewards)
        dones = 1.0 - torch.tensor(dones, dtype=torch.float32)
        v = torch.cat([*v, v_next.view_as(v[0])])

        return (
            rewards.reshape(t, n_agents).to(DEVICE),
            dones.reshape(t, n_agents).to(DEVICE),
            torch.cat(log_probs).reshape(t, n_agents).to(DEVICE),
            v.reshape(t + 1, n_agents).to(DEVICE),
            torch.cat(entropy).reshape(t, n_agents).to(DEVICE),
        )

    def __len__(self):
        return len(self.samples)


class Normalizer:
    """
    Keep rolling mean and std over a distribution.
    Receives numpy array and returns torch tensor
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mean = None
        self.std = None

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(DEVICE)
        else:
            x = torch.tensor(x).float().to(DEVICE)

        # First call init mean and std
        if self.mean == None:
            self.mean = x.mean()
            self.std = x.std()
        else:
            self.mean += self.alpha * (x.mean() - self.mean)
            self.std += self.alpha * (x.std() - self.std)

        # avoid dividing by 0
        return (x - self.mean) / (1e-6 + self.std)
