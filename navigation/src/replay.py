import numpy as np

# Hyperparameters
CAPACITY = int(1e5)
ALPHA = 0.6
BETA_START = 0.4
BETA_STEPS_TO_ONE = 2000


class PrioritizedReplayBuffer:
    def __init__(self, batch_size, capacity=CAPACITY, seed=42):
        """
        Implementation of described improvement on DQN from:
        https://arxiv.org/pdf/1511.05952.pdf

        Keeping track of max_priority avoid O(CAPACITY) operations

        The bottlenecks of this implementation are:
        - O(CAPACITY) probabilities alpha sum, I tried keeping a running sum but updates are unstable
        - O(BATCH_SIZE) sampling and index updating
        - O(CAPACITY) when computing weights.max()
        - O(CAPACITY * 2) memory requirements because we are keeping track of samples and priorities
            separately. Need to explore 'sum-tree' or binary heap like the paper mentions.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.last_sample_indices = None

        # Hyperparams
        self.beta = BETA_START
        self.rng = np.random.RandomState(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Add new experience to memory, assigning max_priority to bias
        the experience to be sampled at least
        """
        experience = (state, action, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.pos] = experience

        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self):
        probs = self._get_probs()
        indices = self.rng.choice(len(self.memory), self.batch_size, p=probs)
        self.last_sample_indices = indices
        weights = self._get_weights(probs[indices])
        samples = [self.memory[i] for i in indices]
        return samples, weights

    def _get_probs(self):
        """
        Compute probability distribution for stored experiences
        P(i) = p_i^(alpha) / sum_k(p_k^(alpha))
        """
        end = max(self.pos, len(self.memory))
        probs = self.priorities[:end] ** ALPHA
        return probs / probs.sum()

    def _get_weights(self, P):
        """
        Compute Importance Sampling (IS)
        w_j = (N * P(j))^(-beta) / max_i(w_i)
        """
        # slowly increase beta to reach 1.0
        self.beta = max(1.0, self.beta + (1.0 - BETA_START) / BETA_STEPS_TO_ONE)
        N = len(self.memory)
        weights = (N * P) ** (-self.beta)
        return weights / weights.max()

    def update_priorities(self, priorities):
        """
        Update priorities after gradient descent using self.last_sample_indices
        p_i = |delta_i| + epsilon
        """
        for i, p in zip(self.last_sample_indices, priorities):
            self.max_priority = max(self.max_priority, p)
            self.priorities[i] = p

    def __len__(self):
        return len(self.memory)
