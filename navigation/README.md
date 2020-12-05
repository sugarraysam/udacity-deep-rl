# Banada collection agent

# Notes

- Rewards and TD error are clipped to fall within [-1,1]
- DQN algorithm + prioritized experience replay
- Baseline vanilla DQN + random uniform policy

# Hyperparameters

| Description            |  Value  |
| ---------------------- | :-----: |
| Minibatch size         |   32    |
| SGD update frequency   |    4    |
| Experience memory size | 100 000 |
| Learning rate          | 0.0005  |
| Prioritization (alpha) |   0.6   |
| Prioritization (beta)  |   0.4   |

# Ressources

- [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)
- [DQN Adventure: from Zero to State of the Art](https://github.com/higgsfield/RL-Adventure)
