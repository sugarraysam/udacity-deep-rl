from control.model import A2CGaussian, DEVICE
from control.minibatch import Sample, MiniBatch, Normalizer

import torch
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

# Hyperparameters
GRADIENT_CLIP = 0.5  # clipping gradients
LR = 1e-3  # learning rate, 10^(-3)
BATCH_SIZE = 4  # update network every N steps


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        n_agents,
        batch_size=BATCH_SIZE,
        grad_clip=GRADIENT_CLIP,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.model = A2CGaussian(state_dim, action_dim)
        self.optimizer = RMSprop(self.model.parameters(), lr=LR)
        self.last_out = None
        self.state_normalizer = Normalizer()
        self.minibatch = MiniBatch()
        self.grad_clip = grad_clip

    def act(self, states):
        """
        Retrieve the actions from the network and save output.
        """
        self.last_out = self.model(self.state_normalizer(states))
        return self.last_out.actions.numpy()

    def step(self, rewards, dones):
        """
        Step through the environment, doing an update if any of the agent reaches a terminal
        state or if we accumulate `batch_size` steps.
        Returns the loss if an update was made, otherwise None.
        """
        _, l, v, e = self.last_out
        if len(self.minibatch) == self.batch_size or (
            any(dones) and len(self.minibatch) > 0
        ):
            # We use v as v_next, and ignore other variables
            return self._learn(v)
        else:
            sample = Sample(rewards=rewards, dones=dones, log_probs=l, v=v, entropy=e)
            self.minibatch.append(sample)
            return None

    def _learn(self, v_next):
        """
        Do a network update, clipping gradients for stability.
        """
        loss = self.minibatch.compute_loss(v_next)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        return loss.item()
