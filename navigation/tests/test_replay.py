from navigation.src.replay import PrioritizedReplayBuffer, ALPHA

import gym
import numpy as np

CARTPOLE_ENV = "CartPole-v1"


def test_replay_integration():
    n = 10
    batch_size = 3
    capacity = 5  # capacity > batch_size
    buffer = PrioritizedReplayBuffer(batch_size, capacity=capacity)
    env = gym.make(CARTPOLE_ENV)
    state = env.reset()

    for i in range(n):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        state = env.reset() if done else next_state

    assert buffer.max_priority == 1.0
    assert len(buffer) == capacity
    assert buffer.pos == (n % capacity)
    assert probabilities_sum_to_one(buffer)

    samples, weights = buffer.sample()
    assert batch_size == len(samples) == len(weights)

    new_priorities = np.random.uniform(0, 1, (batch_size,))
    new_priorities[0] = 1.2
    buffer.update_priorities(new_priorities)

    assert probabilities_sum_to_one(buffer)
    assert buffer.max_priority != 1.0


def probabilities_sum_to_one(buffer):
    probs = buffer.priorities[: len(buffer)] ** ALPHA
    probs /= probs.sum()
    return np.isclose(probs.sum(), 1.0)
