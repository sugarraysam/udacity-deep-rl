from control.model import A2CGaussian, DEVICE
from control.minibatch import Sample, MiniBatch

import torch
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

# Hyperparameters
GRADIENT_CLIP = 0.5  # clipping gradients
LR = 7e-4  # learning rate
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
        self.mb = MiniBatch(n_agents, batch_size)
        self.grad_clip = grad_clip

    def act(self, states):
        """
        Retrieve the actions from the network and save output.
        """
        states = torch.from_numpy(states).float().to(DEVICE)
        self.last_out = self.model(states)
        return self.last_out.actions.numpy()

    def step(self, next_states, rewards, dones):
        """
        Step through the environment, possibly doing an update.
        Returns the loss if an update was made, otherwise None.
        """
        _, l, v, e = self.last_out
        sample = Sample(rewards=rewards, dones=dones, log_probs=l, v=v, entropy=e)
        self.mb.append(sample)

        if len(self.mb) == self.batch_size:
            return self._learn(next_states)

        return None

    def _learn(self, next_states):
        """
        Do a network update, clipping gradients for stability.
        """
        next_states = torch.from_numpy(next_states).float().to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            v_next = self.model.critic(next_states)
        self.model.train()

        loss = self.mb.compute_loss(v_next)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        return loss.item()
