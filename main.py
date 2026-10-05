from dqn_model import DQN
from dqn_pong import Agent
import dqn_pong
import argparse
import hyperparameters as hp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import wrappers
import ExperienceBuffer
from torch.utils.tensorboard.writer import SummaryWriter



class main():

    # The training loop
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--dev", default=" cpu", help="Device name, default = cpu")
        parser.add_argument("--env", default= hp.DEFAULT_ENV_NAME, help="Environment, default = PongNoFrameskip-v4")
        args = parser.parse_args()
        device = torch.device(args.dev)

        # creating the environment using our wrappers
        env = wrappers.create_env(args.env)
        # our NN, which has to be trained
        net = DQN(env.observation_space.shape, env.action_space.n)
        # our target NN
        target_net = DQN(env.observation_space.shape, env.action_space.n)

        writer = SummaryWriter(comment= "-" + args.env)
        print(net)
        # creating our experience replay buffer
        exp_buffer = ExperienceBuffer(hp.REPLAY_SIZE)
        # creating our agent
        agent = Agent(env, exp_buffer)
        # the starting epsilon
        epsilon = hp.EPSILON_START

        # the optimizer
        optimizer = optim.Adam(net.parameters(), lr=hp.LEARNING_RATE)
        total_rewards = []
        frame_counter = 0
        ts_frame = 0 
        ts = time.time()
        best_mean_reward = None

        while True:
            # incrementing the frame counter
            frame_counter += 1
            # epsilon decaying schedule (Epsilon will drop linearly during the given number of frames)
            epsilon = max(hp.EPSILON_FINAL, hp.EPSILON_START - frame_counter/hp.EPSILON_DECAY_LAST_FRAME)
            # making the agent play one step in the environment with our current network and value of epsilon
            # Note: this function returns None if the step is not the final step of the corresponding episode
            reward = agent._step(net, device, epsilon)
            if reward is not None:
                speed = (frame_counter - ts_frame)/(time.time() - ts)
                # appending the total accumulated reward from the whole episode to the  total rewards buffer
                total_rewards.append(reward)
                ts = time.time()
                # mean reward for the last 100 episodes
                mean_reward = np.mean(total_rewards[-100:])
                print(f"frame: {frame_counter}, episode: {len(total_rewards)},speed: {speed}, mean reward: {mean_reward}, epsilon: {epsilon}\n")
                writer.add_scalar("epsilon", epsilon, frame_counter)
                writer.add_scalar("reward", reward, frame_counter)
                writer.add_scalar("mean_reward", mean_reward, frame_counter)
                writer.add_scalar("speed", speed, frame_counter)

                if best_mean_reward is None or mean_reward > best_mean_reward:
                    torch.save(net.state_dict(), f"{args.env}-best{mean_reward:.0f}.dat")
                    if best_mean_reward is not None:
                        print(f"best mean reward updated {best_mean_reward} -> {mean_reward}\n")
                    best_mean_reward = mean_reward

                    if mean_reward > hp.MEAN_REWARD_BOUND:
                        print(f"solved :)\n")
                        print(f"solved in {frame_counter} frames!\n")
                        break

            if len(exp_buffer) < hp.REPLAY_START_SIZE:
                continue
            if frame_counter % hp.SYNC_TARGET_FRAMES == 0:
                # syncing the parameters from the main network to the the target network every SYNC_TARGET_FRAMES frames 
                target_net.load_state_dict(net.state_dict())

            # wee zero the gradients
            optimizer.zero_grad()
            # getting a random sample of batch from the experience replay buffer
            batch = exp_buffer.sample(hp.BATCH_SIZE)
            # calculating loss
            loss_t = dqn_pong.calculate_loss(batch, net, target_net, device)
            loss_t.backward()
            # performing the optimization step to minimize loss
            optimizer.step()
        
        writer.close()





