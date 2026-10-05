# DQN Pong

A small Deep Q-Network (DQN) trainer for Atari Pong. It uses PyTorch for the convolutional Q-network, Gymnasium for environment interaction, Stable-Baselines3 Atari preprocessing, and a replay buffer for sampled training transitions.

## How it works

The environment observations are preprocessed and stacked into four frames. The agent selects actions with an epsilon-greedy policy, stores transitions in replay memory, and trains the online network from random batches. A separate target network supplies next-state values. Training stops when the running mean reward exceeds the configured threshold.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Command-line training entry point, optimization loop, TensorBoard logging, and best-model checkpoint saving. |
| `dqn_model.py` | Convolutional neural network that estimates Q-values for available actions. |
| `dqn_pong.py` | Agent interaction, batch conversion, and DQN loss calculation. |
| `experience.py` | Transition data structure and related type aliases. |
| `ExperienceBuffer.py` | Bounded replay memory with random sampling. |
| `wrappers.py` | Atari preprocessing, channel-first image conversion, and four-frame stacking. |
| `hyperparameters.py` | Environment and training settings. |

## Requirements and setup

Use Python 3 with PyTorch, Gymnasium, Stable-Baselines3, NumPy, and TensorBoard. An Atari-capable Gymnasium/ALE installation and Pong ROMs are also required. See the [Gymnasium Atari setup guide](https://gymnasium.farama.org/environments/atari/) for Atari installation and ROM details.

Install the Python packages in your active virtual environment:

```bash
python -m pip install torch gymnasium stable-baselines3 numpy tensorboard
```

The default environment ID is `PongNoFrameskip-v4`. Environment IDs vary between Gymnasium/ALE versions; if this ID is unavailable, pass an Atari environment ID supported by your installation with `--env`.

## Run training

From the project directory, activate the environment containing the dependencies and run:

```bash
python main.py
```

Select a different environment or device with the optional arguments:

```bash
python main.py --env PongNoFrameskip-v4 --dev cpu
```

For a CUDA-capable PyTorch installation, `--dev cuda` selects the GPU. Training runs until the mean reward crosses the configured threshold. Episode metrics are written to TensorBoard, and improving checkpoints are saved in the project directory as `<environment>-best<reward>.dat`.

To view TensorBoard logs, start TensorBoard in a second terminal:

```bash
tensorboard --logdir runs
```

## Default settings

Values are defined in `hyperparameters.py`:

| Setting | Default |
| --- | ---: |
| Environment | `PongNoFrameskip-v4` |
| Discount factor (`GAMMA`) | `0.99` |
| Batch size | `32` |
| Replay capacity | `10,000` transitions |
| Replay warm-up | `10,000` transitions |
| Learning rate | `0.0001` |
| Target network sync interval | `1,000` frames |
| Epsilon schedule | `1.0` to `0.01` over `150,000` frames |
| Mean reward threshold | `19` |

## Current limitations

- Terminations and time-limit truncations are currently stored together, so the loss treats both as terminal and does not bootstrap from a truncated episode's final observation.
- The target network is synchronized on the configured frame interval. With the defaults, its first sync coincides with the end of replay warm-up.
- The project saves model weights but does not yet provide checkpoint loading or a separate evaluation script.

## License

Released under the MIT License. See [LICENSE](LICENSE).
