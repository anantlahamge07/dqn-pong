import torch 
import gymnasium as gym
import numpy as np
import argparse
import typing as tt

import wrappers
import dqn_model

import collections
import hyperparameters as hp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", help="Environment name, default name: " + hp.DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", required=True, help="Directory for video")
    args = parser.parse_args()

    env = wrappers.create_env(args.env, render_mode = "rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _:stg)
    # loading the learned parameters from the given model
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    counter: tt.Dict[int, int] = collections.Counter()

    while True:
        state_tensor = torch.tensor([state])
        q_values = net(state_tensor).data.numpy()[0]
        action = int(np.argmax(q_values))
        counter[action] += 1
        state, reward, is_done, is_truncated, _ = env.step(action)
        total_reward += reward
        if is_done or is_truncated:
            break

    print(f"Total reward accumulated: {total_reward}")
    print(f"Action counts: {counter}")
    env.close()
