# DQN Pong

A small, from-scratch **Deep Q-Network (DQN)** project for learning to play Atari Pong from pixel observations. The project is written in Python with PyTorch and Gymnasium, and follows the classic DQN approach: a convolutional Q-network, frame stacking, and an experience replay buffer.

> **Project status:** this repository is a work in progress. It contains the model and several training components, but it does not yet include the optimization/target-network training loop or a command-line entry point. As a result, there is not currently a complete command for training an agent end to end.

## What is included

- `dqn_model.py` — convolutional neural network that maps stacked image frames to action-value (Q) estimates.
- `wrappers.py` — Atari preprocessing and observation transforms: channel-first image layout and four-frame stacking.
- `ExperienceBuffer.py` — bounded replay buffer for storing and sampling transitions.
- `dqn_pong.py` — transition data structure and an agent helper that selects actions and steps the environment.
- `hyperparameters.py` — default environment name and DQN hyperparameters.

## How it is intended to work

The agent observes a stack of four preprocessed Pong frames. The DQN estimates a Q-value for each available action; during interaction, the agent uses epsilon-greedy action selection to balance exploration and exploitation. Each transition is added to replay memory, from which batches can be sampled for learning.

The configured defaults include a replay capacity of 10,000 transitions, a batch size of 32, a discount factor of 0.99, and a linearly decaying exploration rate. The training loop that would use these settings has not been implemented yet.

## Requirements

The source imports the following packages:

- Python 3
- [PyTorch](https://pytorch.org/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (used here for its Atari wrapper utilities)
- NumPy
- TensorBoard (imported by the agent module)

Install the Python packages in your environment with:

```bash
python -m pip install torch gymnasium stable-baselines3 numpy tensorboard
```

Atari environments also require the appropriate Atari/ALE support and ROMs for your Gymnasium setup. Consult the [Gymnasium Atari documentation](https://gymnasium.farama.org/environments/atari/) for installation instructions. The configured environment ID is `PongNoFrameskip-v4`; environment IDs and Atari setup can vary by Gymnasium/ALE version.

## Running

There is no end-to-end training command at this stage. `dqn_pong.py` currently defines helper classes only; it does not parse arguments, create the environment, run optimization, save a model, or launch evaluation. A training loop and an executable entry point are needed before the agent can be trained from the command line.

## Configuration

Defaults are defined in `hyperparameters.py`:

| Setting | Default | Purpose |
| --- | ---: | --- |
| Environment | `PongNoFrameskip-v4` | Atari Pong environment ID |
| Discount factor (`GAMMA`) | `0.99` | Future reward discount |
| Batch size | `32` | Transitions sampled per update |
| Replay capacity | `10,000` | Maximum transitions retained |
| Learning rate | `0.0001` | Optimizer learning rate |
| Replay start size | `10,000` | Intended warm-up before training |
| Target sync interval | `1,000` frames | Intended target-network update frequency |
| Epsilon | `1.0` to `0.01` | Exploration schedule over 150,000 frames |

These include settings for a planned training loop; not all are consumed by the current code.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
