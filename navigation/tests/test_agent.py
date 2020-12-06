from navigation.src.agent import Agent, EPSILON_START

import gym
import os
import numpy as np
from copy import deepcopy

CARTPOLE_ENV = "CartPole-v1"
CHECKPOINT = "tests/checkpoint.pth"


def test_agent_integration():
    batch_size = 4
    n_steps = 12

    env = gym.make(CARTPOLE_ENV)
    agent = Agent(
        env.observation_space.shape[0], env.action_space.n, batch_size=batch_size
    )

    initial_state = deepcopy(agent.dqn_local.state_dict())

    state = env.reset()
    for i in range(n_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        loss = agent.step(state, action, reward, next_state, done)
        assert loss is None or type(loss) == float
        state = env.reset() if done else next_state

    assert agent.epsilon != EPSILON_START
    assert len(agent.memory) == n_steps
    assert not state_dicts_are_equal(initial_state, agent.dqn_local.state_dict())

    agent.dqn_local.checkpoint(CHECKPOINT)
    assert os.path.exists(CHECKPOINT)
    os.remove(CHECKPOINT)
    assert not os.path.exists(CHECKPOINT)


def state_dicts_are_equal(sd1, sd2):
    for k in sd1.keys():
        if sd1[k].data.ne(sd2[k].data).sum() > 0:
            return False
    return True
