"""Training script for a DQN agent in the simple Minecraft-like environment."""

import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import DQNAgent
from env import DummyMinecraftEnv
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Configuration & Reproducibility
# ---------------------------------------------------------------------------
MODEL_DIR = "trained_models"
LOAD_PATH = os.path.join(MODEL_DIR, "model_ep8000.pth")
WIN_SAVE_PREFIX = "model_win_ep"
MISTAKE_MARGIN = 0.05
EPISODES = 100_000

BUFFER_SIZE = 10_000
BATCH_SIZE = 64
GAMMA = 0.99

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY_EPISODES = 50_000

EVAL_INTERVAL = 1000
EVAL_EPISODES = 10

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
print(f"Using seed {SEED}")


def epsilon_by_episode(ep: int) -> float:
    """Linear epsilon decay."""
    if ep >= EPS_DECAY_EPISODES:
        return EPS_END
    slope = (EPS_END - EPS_START) / EPS_DECAY_EPISODES
    return EPS_START + slope * (ep - 1)


def evaluate(agent: DQNAgent, episodes: int) -> tuple[float, float, float, float]:
    """Run evaluation episodes without exploration."""
    env = DummyMinecraftEnv()

    rewards: list[float] = []
    steps_list: list[int] = []
    regrets: list[float] = []
    wins = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        ep_regrets: list[float] = []

        while not done:
            q_values = agent.predict_q(state)
            action = int(np.argmax(q_values))
            q_sa = float(q_values[action])
            max_q = float(np.max(q_values))
            ep_regrets.append(max_q - q_sa)

            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)
        regrets.append(np.mean(ep_regrets) if ep_regrets else 0.0)
        if reward > 0:
            wins += 1

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    win_rate = wins / episodes if episodes else 0.0
    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
    avg_regret = float(np.mean(regrets)) if regrets else 0.0
    return avg_reward, win_rate, avg_steps, avg_regret


def train() -> None:
    env = DummyMinecraftEnv()
    state_dim = len(env.reset())
    action_dim = 4

    agent = DQNAgent(state_dim, action_dim, gamma=GAMMA)
    agent.load(LOAD_PATH)
    buffer = ReplayBuffer(BUFFER_SIZE)

    os.makedirs(MODEL_DIR, exist_ok=True)

    rewards: list[float] = []

    train_csv = "train_metrics.csv"
    eval_csv = "eval_metrics.csv"
    if not os.path.exists(train_csv):
        with open(train_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "episode",
                    "total_reward",
                    "steps",
                    "avg_td_error",
                    "avg_regret",
                    "mistakes",
                    "epsilon",
                    "win",
                ]
            )
    if not os.path.exists(eval_csv):
        with open(eval_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "avg_reward", "win_rate", "avg_steps", "avg_regret"]
            )

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        mistakes = 0
        regrets: list[float] = []
        td_errors: list[float] = []

        epsilon = epsilon_by_episode(ep)

        while not done:
            q_values = agent.predict_q(state)
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                action = int(np.argmax(q_values))

            q_sa = float(q_values[action])
            max_q = float(np.max(q_values))
            regret = max_q - q_sa
            regrets.append(regret)
            if q_sa < max_q - MISTAKE_MARGIN:
                mistakes += 1

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                td_error = agent.train_step(batch)
                td_errors.append(td_error)

        avg_td_error = float(np.mean(td_errors)) if td_errors else 0.0
        avg_regret = float(np.mean(regrets)) if regrets else 0.0
        win = int(reward > 0)
        if win:
            agent.save(os.path.join(MODEL_DIR, f"{WIN_SAVE_PREFIX}{ep}.pth"))

        rewards.append(total_reward)
        with open(train_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, total_reward, steps, avg_td_error, avg_regret, mistakes, epsilon, win]
            )

        if ep % EVAL_INTERVAL == 0:
            avg_reward_last = float(np.mean(rewards[-EVAL_INTERVAL:]))
            eval_avg_reward, eval_win_rate, eval_avg_steps, eval_avg_regret = evaluate(
                agent, EVAL_EPISODES
            )
            print(
                f"Episode {ep} - Avg Reward: {avg_reward_last:.3f} "
                f"Avg TD Error: {avg_td_error:.3f} Avg Regret: {avg_regret:.3f} "
                f"Mistakes: {mistakes} Epsilon: {epsilon:.3f} "
                f"Eval Avg Reward: {eval_avg_reward:.3f} Eval Win Rate: {eval_win_rate:.2f} "
                f"Eval Avg Steps: {eval_avg_steps:.2f} Eval Avg Regret: {eval_avg_regret:.3f}"
            )
            agent.update_target()
            agent.save(os.path.join(MODEL_DIR, f"model_ep{ep}.pth"))
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [ep, eval_avg_reward, eval_win_rate, eval_avg_steps, eval_avg_regret]
                )

    agent.save(os.path.join(MODEL_DIR, "model_final.pth"))

    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Rewards")
    plt.savefig("reward_plot.png")


if __name__ == "__main__":
    train()

