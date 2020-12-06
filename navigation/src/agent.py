from navigation.src.model import DQN
from navigation.src.replay import PrioritizedReplayBuffer

import numpy as np
import warnings

import torch
import torch.optim as optim

# Hyperparameters
BATCH_SIZE = 32  # minibatch size
GAMMA = 0.99  # discount factor
LR = 5e-4  # learning rate
SGD_FREQ = 4  # frequency of SGD updates
TARGET_UPDATE_FREQ = 1000  # frequency of dqn_target parameters update
PRIORITIES_EPSILON = 1e-5  # small value added to priorities to avoid no sampling
EPSILON_START = 1.0  # For epsilon greedy policy
EPSILON_DECAY = 0.9995  # Decay rate for epsilon greedy policy
EPSILON_FINAL = 0.01  # For epsilon greedy policy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        batch_size=BATCH_SIZE,
        epsilon=EPSILON_START,
        seed=42,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.n_steps = 0
        self.rng = np.random.RandomState(seed)
        self.epsilon = epsilon

        # DQN
        self.dqn_local = DQN(state_size, action_size, seed=seed).to(DEVICE)
        self.dqn_target = DQN(state_size, action_size, seed=seed).to(DEVICE)
        self.optimizer = optim.Adam(self.dqn_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(batch_size)

    def step(self, state, action, reward, next_state, done):
        """
        Step through the environment, possibly doing an SGD update
        """
        self.memory.add(state, action, reward, next_state, done)
        self.n_steps += 1
        loss = None
        if (self.n_steps % SGD_FREQ) == 0 and len(self.memory) > self.batch_size:
            samples, weights = self.memory.sample()
            mb = MiniBatch(samples, weights)
            mb.to_torch()
            loss = self.learn(mb)
        return loss

    def learn(self, mb):
        """
        Perform SGD update on self.dqn_local parameters using a minibatch
        sampled from the PrioritizedReplayBuffer.

        Loss function is mean squared error /w Importance Sampling

        Update sampling priorities of the sampled experiences
        """
        q_values = self.dqn_local(mb.states).gather(1, mb.actions)
        next_q_values = self.dqn_target(mb.next_states).detach().max(1)[0].unsqueeze(1)
        expected_q_values = mb.rewards + (GAMMA * next_q_values * (1 - mb.dones))

        loss = (q_values - expected_q_values).pow(2) * mb.weights
        new_priorities = loss + PRIORITIES_EPSILON
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(new_priorities.data.cpu().numpy())
        self.optimizer.step()

        if (self.n_steps % TARGET_UPDATE_FREQ) == 0:
            self.update_target_params()

        return loss.item()

    def update_target_params(self):
        """
        Updates self.dqn_target weights to be in sync with self.dqn_local
        """
        self.dqn_target.load_state_dict(self.dqn_local.state_dict())

    def act(self, state):
        """
        Select next action according to an epsilon-greedy policy
        """
        state = (
            torch.from_numpy(np.fromiter(state, dtype=np.float32))
            .unsqueeze(0)
            .to(DEVICE)
        )
        self.dqn_local.eval()
        with torch.no_grad():
            action_values = self.dqn_local(state)
        self.dqn_local.train()

        # Epsilon-greedy action selection
        self._decay_epsilon()
        if self.rng.uniform(0, 1) > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return self.rng.choice(np.arange(self.action_size))

    def _decay_epsilon(self):
        """
        Slowly decay epsilon with each training steps
        """
        self.epsilon = max(EPSILON_FINAL, self.epsilon * EPSILON_DECAY)


class MiniBatch:
    def __init__(self, samples, weights):
        states, actions, rewards, next_states, dones = zip(*samples)
        self.states = np.vstack(states).astype(np.float32)
        self.actions = np.fromiter(actions, dtype=np.long)
        self.rewards = np.fromiter(rewards, dtype=np.float32)
        self.next_states = np.vstack(next_states).astype(np.float32)
        self.dones = np.fromiter(dones, dtype=np.float32)
        self.weights = weights.astype(np.float32)

    def to_torch(self):
        self.states = torch.from_numpy(self.states).to(DEVICE)
        self.actions = torch.from_numpy(self.actions).reshape(-1, 1).to(DEVICE)
        self.rewards = torch.from_numpy(self.rewards).reshape(-1, 1).to(DEVICE)
        self.next_states = torch.from_numpy(self.next_states).to(DEVICE)
        self.dones = torch.from_numpy(self.dones).reshape(-1, 1).to(DEVICE)
        self.weights = torch.from_numpy(self.weights).reshape(-1, 1).to(DEVICE)
