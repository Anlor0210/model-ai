"""Training script for a DQN agent in the simple Minecraft-like environment."""

import matplotlib.pyplot as plt
import numpy as np

from agent import DQNAgent
from env import DummyMinecraftEnv
from replay_buffer import ReplayBuffer


EPISODES = 100_000
BUFFER_SIZE = 10_000
BATCH_SIZE = 64
GAMMA = 0.99

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY_EPISODES = 50_000


def epsilon_by_episode(ep: int) -> float:
    if ep >= EPS_DECAY_EPISODES:
        return EPS_END
    slope = (EPS_END - EPS_START) / EPS_DECAY_EPISODES
    return EPS_START + slope * (ep - 1)


def train() -> None:
    env = DummyMinecraftEnv()
    state_dim = len(env.reset())
    action_dim = 4

    agent = DQNAgent(state_dim, action_dim, gamma=GAMMA)
    buffer = ReplayBuffer(BUFFER_SIZE)

    rewards: list[float] = []

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        epsilon = epsilon_by_episode(ep)

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                agent.train_step(batch)

        rewards.append(total_reward)

        # Periodic updates and logging
        if ep % 1000 == 0:
            avg_reward = np.mean(rewards[-1000:])
            print(f"Episode {ep} - Avg Reward: {avg_reward:.3f}")
            agent.update_target()
            agent.save(f"model_ep{ep}.pth")

    # Save final model
    agent.save("model_final.pth")

    # Plot rewards over episodes
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Rewards")
    plt.savefig("reward_plot.png")


if __name__ == "__main__":
    train()

