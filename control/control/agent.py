from control.model import A2CGaussian, DEVICE
from control.minibatch import Sample, MiniBatch

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
        self.state_normalizer = StateNormalizer()
        self.minibatch = MiniBatch()
        self.grad_clip = grad_clip

    def act(self, states):
        """
        Retrieve the actions from the network and save output.
        """
        self.last_out = self.model(self.state_normalizer(states))
        return self.last_out.actions.numpy()

    def step(self, next_states, rewards, dones):
        """
        Step through the environment, doing an update if any of the agent reaches a terminal
        state or if we accumulate `batch_size` steps.
        Returns the loss if an update was made, otherwise None.
        """
        _, l, v, e = self.last_out
        sample = Sample(rewards=rewards, dones=dones, log_probs=l, v=v, entropy=e)
        self.minibatch.append(sample)

        if len(self.minibatch) == self.batch_size or any(dones):
            return self._learn(next_states)
        else:
            return None

    def _learn(self, next_states):
        """
        Do a network update, clipping gradients for stability.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.state_normalizer(next_states))
        self.model.train()

        loss = self.minibatch.compute_loss(out.v)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        return loss.item()


class StateNormalizer:
    """
    Keep rolling mean and std over states distribution.
    Receives numpy array and returns torch tensor
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mean = None
        self.std = None

    def __call__(self, x):
        x = torch.from_numpy(x).float().to(DEVICE)
        # First call init mean and std
        if self.mean == None:
            self.mean = x.mean()
            self.std = x.std()
        else:
            self.mean += self.alpha * (x.mean() - self.mean)
            self.std += self.alpha * (x.std() - self.std)

        # avoid dividing by 0
        return (x - self.mean) / (1e-6 + self.std)
