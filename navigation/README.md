# Banada collection agent <!-- omit in toc -->

- [Project](#project)
  - [Environment](#environment)
- [Installation](#installation)
- [Tests](#tests)
- [Report](#report)
  - [Hyperparameters for `agent.Agent`](#hyperparameters-for-agentagent)
  - [Hyperparameters for `replay.PrioritizedReplayBuffer`](#hyperparameters-for-replayprioritizedreplaybuffer)
  - [Hyperparameters for `model.DQN`](#hyperparameters-for-modeldqn)
- [Ideas for the future](#ideas-for-the-future)
- [Resources](#resources)

# Project

## Environment

We are solving the Banana environment. The task is episodic, and to solve it, the agent must get an average score of +13 over 100 consecutive episodes.

**Rewards**

+1 for collecting a yellow banana
-1 for collecting a blue banana

**State**

37 dimensions, containing the agent's velocity, along with ray-based perception of objects around the agent's forward direction

**Actions**

- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right


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
$ cd navigation
$ ./run_tests.sh
```

# Report

I decided to solve this environment using the Deep-Q-Learning algorithm with Prioritized Replay Buffer. I set the episode limit to 2000, but I am not limiting the number of timesteps per episode because the rewards are received frequently enough to ensure proper learning.

A couple of interesting points about the DQN algorithm:
- It computes the target `q_values` using a second network implementation, with stable parameters, that are updated according to the `Target update frequency`
- We do a Stochastic Gradient Descent to update the local dqn network params following the `SGD update frequency`
- The loss function is the mean squared error (MSE) with Importance Sampling
- Experience tuples are stored on each step in the PrioritizedReplayBuffer (state, action, reward, next_state, done)
- Minibatches are sampled from the PrioritizedReplayBuffer, using a probability distribution built from priorities associated to each experience
- Priorities associated to each experience are equal to the loss plus a small epsilon to avoid having a 0% chance of sampling that experience, so priorities are greater than 0
- The Agent follows an epsilon-greedy policy, with epsilon (degree of exploration) decaying after each time step, until it reaches a final stable value
- We use the `Adam` optimizer to take advantage of momentum, which allows SGD to *hopefully* not get stuck and reach a local minima
- The `alpha` hyperparameter from the `PrioritizedReplayBuffer` gives a *boost* to experiences with low priorities, and penalizes high priority values, to make sampling more uniform (while still being prioritized)
- The `DQN Neural Network` structure is `37 x 64 x 64 x 4`. Where `37` is the `state_size` and `4` represents the `q_values` associated with each possible action. Every layer is fully connected, with the Relu activation function.

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

There are a few extensions to vanilla DQN, that improve and push the boundaries of what is the state-of-the-art DQN implmentation. In the future, it would be important to understand and implement each of those extensions. They are all listed on this great resource I found:

- [DQN Adventure: from Zero to State of the Art](https://github.com/higgsfield/RL-Adventure)

# Resources

- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
- [DQN Adventure: from Zero to State of the Art](https://github.com/higgsfield/RL-Adventure)
