from control.agent import Agent

import gym
import os
import torch
import numpy as np
from copy import deepcopy


def test_agent_single_step():
    n_agents = 4
    env = DummyEnv(n_agents)
    state_dim, action_dim = env.dimensions()
    agent = Agent(state_dim, action_dim, n_agents)

    states = env.reset()
    actions = agent.act(states)
    assert actions_are_valid(actions)
    assert actions.shape == (n_agents, action_dim)

    next_states, rewards, dones = env.step(actions)

    assert agent.last_out is not None
    assert len(agent.minibatch) == 0

    loss = agent.step(next_states, rewards, dones)

    assert loss is None
    assert len(agent.minibatch) == 1


def test_agent_single_batch():
    batch_size, n_agents = 4, 4
    env = DummyEnv(n_agents)
    state_dim, action_dim = env.dimensions()
    agent = Agent(state_dim, action_dim, n_agents, batch_size=batch_size)
    initial_params = deepcopy(agent.model.state_dict())

    states = env.reset()
    for i in range(batch_size):
        actions = agent.act(states)
        next_states, rewards, dones = env.step(actions)
        _ = agent.step(next_states, rewards, dones)
        states = env.reset() if dones.any() else next_states

    assert not params_are_equal(initial_params, agent.model.state_dict())


def test_agent_multiple_batch():
    batch_size, n_agents = 4, 12
    t = batch_size * 20
    env = DummyEnv(n_agents)
    state_dim, action_dim = env.dimensions()
    agent = Agent(state_dim, action_dim, n_agents, batch_size=batch_size)
    losses = []

    last_params = deepcopy(agent.model.state_dict())
    states = env.reset()
    for i in range(t):
        actions = agent.act(states)
        next_states, rewards, dones = env.step(actions)
        loss = agent.step(next_states, rewards, dones)
        states = env.reset() if any(dones) else next_states
        if i > 0 and len(agent.minibatch) == 0:
            assert loss is not None and not isinstance(loss, torch.Tensor)
            losses.append(loss)
            assert not params_are_equal(last_params, agent.model.state_dict())
            last_params = deepcopy(agent.model.state_dict())

    assert len(losses) == (t // batch_size)

def test_gradient_is_stable():
    # TODO: use pdb, debug pytorch graph, see what is going on
    assert True == False

CHECKPOINT = "control/tests/checkpoint.pth"


def test_model_checkpoint():
    state_dim, action_dim = 33, 4
    n_agents = 20
    agent = Agent(state_dim, action_dim, n_agents)
    agent.model.checkpoint(CHECKPOINT)
    assert os.path.exists(CHECKPOINT)
    os.remove(CHECKPOINT)
    assert not os.path.exists(CHECKPOINT)


def params_are_equal(sd1, sd2):
    for k in sd1.keys():
        if sd1[k].data.ne(sd2[k].data).sum() > 0:
            return False
    return True


class DummyEnv:
    """
    Dummy parallel env for unit testing
    """

    def __init__(self, n_agents, env_name="LunarLanderContinuous-v2"):
        self.n_agents = n_agents
        self.envs = [gym.make(env_name) for _ in range(n_agents)]

    def dimensions(self):
        state_dim = self.envs[0].observation_space.shape[0]
        action_dim = self.envs[0].action_space.shape[0]
        return state_dim, action_dim

    def reset(self):
        states = []
        for env in self.envs:
            states.append(env.reset())
        return np.vstack(states)

    def step(self, actions):
        """
        Sample action from corresponding env
        """
        next_states, rewards, dones = [], [], []
        for action, env in zip(actions, self.envs):
            s, r, d, _ = env.step(action)
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
        return np.vstack(next_states), np.vstack(rewards), np.vstack(dones)


def actions_are_valid(actions):
    return isinstance(actions, np.ndarray) and (np.abs(actions) > 1.0).sum() == 0


def state_dicts_are_equal(sd1, sd2):
    for k in sd1.keys():
        if sd1[k].data.ne(sd2[k].data).sum() > 0:
            return False
    return True
