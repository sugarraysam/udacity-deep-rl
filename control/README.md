# Continuous Control agent <!-- omit in toc -->

- [Project](#project)
  - [Environment](#environment)
- [Installation](#installation)
- [Tests](#tests)
- [Training the agent](#training-the-agent)
- [Report](#report)
  - [Hyperparameters for `agent.Agent`](#hyperparameters-for-agentagent)
  - [Hyperparameters for `replay.PrioritizedReplayBuffer`](#hyperparameters-for-replayprioritizedreplaybuffer)
  - [Hyperparameters for `model.DQN`](#hyperparameters-for-modeldqn)
- [Ideas for the future](#ideas-for-the-future)
- [Resources](#resources)

# Project

## Environment

We are solving the Reacher environment. The agent's goal is to move a double-jointed arm to a location and keep it there. I will be solving the *Option 2 - Distributed Training* version of the environment, where 20 identical agents have their own copy of the environment.

To solve this problem, the agents must get an average score of +30, over 100 consecutive episodes, and over all agents. Specifically:

- After each episode, add the undiscounted rewards for all agents (separately)
- Compute the average over those sums as the final score for that episode

**Rewards**

+0.1 for each step that the agent's hand is in the goal location

**State - Continuous**

33 variables corresponding to position, rotation, velocity, and angular velocities of the arm

**Actions - Continuous**

Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.


# Installation

The project is setup to work as a nested sub-project to the global repo. To setup everything:

```bash
$ git clone git@github.com:sugarraysam/udacity-deep-rl.git
$ cd udacity-deep-rl

# Install main dependencies
$ make pipenv

# pull udacity deep rl repo as git submodule
# to install required dependencies for unityagent environment
$ git submodule update --init
$ pipenv shell
$ cd deep-reinforcement-learning/python
$ pip install .
```

Finally, you need to setup the Unity Banana environment for your machine. Please see the [instructions](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation).

# Tests

You can run all tests using pytest:

```bash
$ pipenv shell
$ cd control
$ ./run_tests.sh
```

# Training the agent

The training code lives inside the jupyter notebook `Navigation.ipynb`. You can run the notebook server with the appropriate python version and packages:

```bash
# Launch jupyter notebook server
$ make jupyter
```

After, navigate to `Navigation.ipynb` inside your browser. You can see the output of the previously ran code. If you've setup everything correctly, you can go ahead and rerun the cells, to see a live training of the agent.

The code should work on GPU as well, although I have not tested this.

# Report

- Read more actor critic implementations + paper
- Read GAE paper + implementations

- Actor network, 64x32x16x4, finish /w relu,relu,relu,tanh
  - how to generate 4 continuous values and replace log_pi(a) in loss function?
  - look @ blog post, use gaussian output?
- Critic network (DQN), learn value function (not state-action), 64x64x1
- distributed training, 20 identical agents, each with its own copy of the environment
- Classes
  - agent.Agent (20x)
  - model.ActorCritic
  - model.Critic
  - model.Minibatch (size 20 == n parallel agents)
- setup.py - create python pkg
- A2C algorithm
- Generalized Advantage Estimation
- Normalized Rewards, clip rewards?

## Hyperparameters for `agent.Agent`

| Description                       | Value  |
| --------------------------------- | :----: |
| Minibatch size                    |   32   |
| Gamma (discount factor)           |  0.99  |
| Learning Rate                     |  5e-4  |
| SGD update frequency              |   4    |
| Target update frequency           |  1000  |
| Priorities epsilon                |  1e-5  |
| Epsilon start                     |  1e-5  |
| Epsilon decay rate (per timestep) | 0.9995 |
| Epsilon final (1% exploration)    |  0.01  |

## Hyperparameters for `replay.PrioritizedReplayBuffer`

| Description                  | Value  |
| ---------------------------- | :----: |
| Capacity                     | 100000 |
| Alpha for prioritization     |  0.6   |
| Beta for Importance Sampling |  0.4   |

## Hyperparameters for `model.DQN`

| Description        |  Value  |
| ------------------ | :-----: |
| Hidden layer sizes | [64,64] |


# Ideas for the future

TODO

# Resources

- [Reader ML Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783v2)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [ShangtongZhang/DeepRL](https://github.com/ShangtongZhang/DeepRL)
- [Medium: Advantage Actor Critic continuous case implementation](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-continuous-case-implementation-f55ce5da6b4c)
