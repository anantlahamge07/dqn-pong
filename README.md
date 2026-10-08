# DQN Pong

A PyTorch Deep Q-Network (DQN) that learns to play Atari Pong through Gymnasium. The project includes a training loop, a convolutional Q-network, experience replay, Atari preprocessing, and a script that runs a saved model while recording a video.

## How it works

The environment wrapper applies Stable-Baselines3 Atari preprocessing, converts image observations to channel-first format, and stacks four frames. The agent chooses actions with an epsilon-greedy policy and stores each transition in a bounded replay buffer. Once the buffer reaches its warm-up size, the training loop samples random batches and minimizes the Bellman error. A target network supplies next-state values and is synchronized periodically. Training stops when the mean reward over the latest 100 episodes exceeds the configured reward bound.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Training entry point, optimization loop, TensorBoard metrics, and best-mean-reward checkpoints. |
| `pong_play.py` | Loads a checkpoint, plays one episode, prints its reward and action counts, and records video. |
| `dqn_model.py` | Convolutional neural network that estimates Q-values for each action. |
| `dqn_pong.py` | Agent interaction, batch conversion, and DQN loss calculation. |
| `experience.py` | Transition data structure and related type aliases. |
| `ExperienceBuffer.py` | Bounded replay memory with random sampling. |
| `wrappers.py` | Atari preprocessing, channel-first image conversion, and four-frame stacking. |
| `hyperparameters.py` | Environment and training settings. |

## Requirements and setup

Use Python 3 with PyTorch, Gymnasium's Atari support, Stable-Baselines3, NumPy, and TensorBoard. Atari environments also need the Arcade Learning Environment and Pong ROMs. The exact environment ID can vary with the Gymnasium/ALE version; the default in this project is `PongNoFrameskip-v4`.

Install the dependencies in a virtual environment. For example:

```bash
python -m pip install torch "gymnasium[atari]" stable-baselines3 numpy tensorboard
```

Install or make the Atari ROMs available according to the [Gymnasium Atari setup guide](https://gymnasium.farama.org/environments/atari/). If the default environment ID is not registered in your installation, use an Atari Pong ID supported by your installed Gymnasium and ALE packages with `--env`.

There is no dependency lockfile or `requirements.txt` in this repository.

## Train

From the project directory, run:

```bash
python main.py
```

The optional command-line arguments select the environment and PyTorch device:

```bash
python main.py --env PongNoFrameskip-v4 --dev cpu
```

Use `--dev cuda` with a CUDA-enabled PyTorch installation to train on an available GPU. Training continues until the latest-100-episode mean reward exceeds `MEAN_REWARD_BOUND`. Episode reward, mean reward, epsilon, and frame speed are logged to TensorBoard. The script writes each new best checkpoint to the project root using the name `<environment>-best<mean-reward-rounded>.dat`.

To inspect the logs in another terminal:

```bash
tensorboard --logdir runs
```

## Play a saved model and record a video

`pong_play.py` runs one episode using greedy actions and writes a Gymnasium video to the directory passed with `--record`:

```bash
python pong_play.py --model PongNoFrameskip-v4-best19.dat --record Video
```

To record a different saved training stage, replace the number after `best` in the model filename. For example, use `PongNoFrameskip-v4-best-21.dat` through `PongNoFrameskip-v4-best19.dat` (where `x` ranges from `-21` to `19`) if those checkpoint files are present:

```bash
python pong_play.py --model PongNoFrameskip-v4-best-8.dat --record Video
```

Each checkpoint contains the model weights saved at that point in training, so the selected file determines which model plays the episode. You can also choose the environment with `--env`. The model file must match the network's environment observation shape and action count. The script prints the episode's total reward and action counts when the episode ends.

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
| Mean reward threshold | `19` over the latest 100 episodes |

## Notes and limitations

- The replay buffer stores termination and truncation in one `done_trunc` flag. The loss therefore treats either event as terminal and does not bootstrap from the final observation of a time-limit truncation.
- The target network starts with its own initial weights and is first synchronized when the frame counter reaches the configured sync interval, provided replay warm-up has completed.
- Checkpoints contain only the online network weights. The optimizer, replay buffer, frame counter, and training state are not saved, so training cannot be resumed from a checkpoint.
- `pong_play.py` evaluates one episode and records it; it does not calculate multi-episode evaluation statistics.

## License

Released under the MIT License. See [LICENSE](LICENSE).
