# DQN Pong

A small, from-scratch **Deep Q-Network (DQN)** project for Atari Pong. It uses PyTorch for the convolutional Q-network and Gymnasium for the environment, with Atari preprocessing, stacked image observations, and an experience replay buffer.

> **Status: work in progress.** The repository contains core DQN components and a partial loss calculation, but the training loop and executable entry point are not complete. Several implementation issues also need to be fixed before the current code can train end to end.

## Project contents

| File | Purpose |
| --- | --- |
| `dqn_model.py` | Convolutional Q-network that predicts a value for each available action from image observations. |
| `wrappers.py` | Atari preprocessing, conversion to channel-first image layout, and four-frame observation stacking. |
| `ExperienceBuffer.py` | Bounded replay memory with random transition sampling. |
| `dqn_pong.py` | `Experience` data structure, epsilon-greedy environment interaction, batch-to-tensor conversion, and a partial DQN loss helper. |
| `hyperparameters.py` | Environment ID and default replay, optimization, target-sync, and exploration settings. |

## DQN outline

The intended agent observes four stacked frames and uses a convolutional network to estimate action values. It selects actions with an epsilon-greedy policy, stores transitions in replay memory, and is designed to learn from random batches while using a target network for next-state values. Hyperparameters for this process are defined in `hyperparameters.py`.

The repository does not yet connect these pieces in a training loop. It also does not currently provide checkpoint saving/loading or an evaluation script.

## Requirements

The source imports:

- Python 3
- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for Atari wrapper utilities
- NumPy
- TensorBoard

Install the Python dependencies with:

```bash
python -m pip install torch gymnasium stable-baselines3 numpy tensorboard
```

An Atari-capable Gymnasium installation and the required ROMs are also needed to create the Pong environment. Follow the [Gymnasium Atari setup guide](https://gymnasium.farama.org/environments/atari/). The configured environment ID is `PongNoFrameskip-v4`; the available ID may differ across Gymnasium/ALE versions.

## Running

There is no supported end-to-end run command yet. Running `dqn_pong.py` does not start training: its `__main__` block is currently empty. Before use as a trainer, the project needs a complete training loop and fixes to the current partial implementation, including the replay-buffer import dependency and the loss/episode-reward calculations.

## Default configuration

Values in `hyperparameters.py`:

| Setting | Default |
| --- | ---: |
| Environment | `PongNoFrameskip-v4` |
| Discount factor (`GAMMA`) | `0.99` |
| Batch size | `32` |
| Replay capacity | `10,000` transitions |
| Learning rate | `0.0001` |
| Replay warm-up size | `10,000` transitions |
| Target network sync interval | `1,000` frames |
| Epsilon schedule | `1.0` to `0.01` over `150,000` frames |
| Mean-reward threshold | `19` |

These settings describe the planned training process; not all are used by the current code yet.

## License

Released under the MIT License. See [LICENSE](LICENSE).
