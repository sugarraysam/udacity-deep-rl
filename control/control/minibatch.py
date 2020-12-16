from control.model import DEVICE
from collections import namedtuple

import torch

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

    def append(self, sample):
        self.samples.append(sample)

    def compute_loss(self, v_next):
        """
        Compute loss using the accumulated samples. Reset stored samples.
        All tensors are of shape (batch_size, n_agents) except for v, because we appended v_next.
        """
        rewards, dones, log_probs, v, entropy = self._parse_samples(v_next)
        self.samples = []
        advantages, returns = self._compute_advantages_and_returns(rewards, dones, v)

        # print(f"v_next_mean: {v_next.mean()}, v_mean: {v[:-1].mean()}")
        # print(f"advantages_mean: {advantages.mean()}")
        # print(f"returns_mean: {returns.mean()}")

        a_loss = -(log_probs * advantages).mean()
        c_loss = 0.5 * (returns - v[:-1]).pow(2).mean()
        e_loss = entropy.mean()
        return a_loss - self.e_weight * e_loss + self.c_weight * c_loss

    def _parse_samples(self, v_next):
        """
        Prepare parameters as torch tensors. Rewards and dones are plain python lists.
        Dones are 0 if true, and 1 if false, so we can use them in equations.
        """
        rewards, dones, log_probs, v, entropy = zip(*self.samples)
        batch_size, n_agents = len(rewards), len(rewards[0])

        rewards = self._normalize_rewards(torch.tensor(rewards, dtype=torch.float32))
        dones = 1.0 - torch.tensor(dones, dtype=torch.float32)
        log_probs = torch.cat(log_probs)
        v = torch.cat([*v, v_next.view_as(v[0])])
        entropy = torch.cat(entropy)

        return (
            rewards.reshape(batch_size, n_agents).to(DEVICE),
            dones.reshape(batch_size, n_agents).to(DEVICE),
            log_probs.reshape(batch_size, n_agents).to(DEVICE),
            v.reshape(batch_size + 1, n_agents).to(DEVICE),
            entropy.reshape(batch_size, n_agents).to(DEVICE),
        )

    def _compute_advantages_and_returns(self, rewards, dones, v):
        batch_size, n_agents = rewards.shape[0], rewards.shape[1]
        all_advantages, all_returns = [], []

        advantages = torch.zeros(n_agents)
        returns = v[-1].clone()
        for i in reversed(range(batch_size)):
            returns = rewards[i] + self.discount * dones[i] * returns
            td_error = rewards[i] + self.discount * dones[i] * v[i + 1] - v[i]
            advantages = advantages * self.tau * self.discount * dones[i] + td_error

            all_advantages.append(advantages.unsqueeze(0).detach())
            all_returns.append(returns.unsqueeze(0).detach())

        # reverse and return as tensors
        return torch.cat(all_advantages[::-1]), torch.cat(all_returns[::-1])

    def _normalize_rewards(self, rewards):
        """
        Update rolling mean and std, avoid dividing by zero
        """
        return (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    def __len__(self):
        return len(self.samples)
