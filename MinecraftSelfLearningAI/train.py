"""Training script for an enhanced DQN agent in a simple environment."""

import csv
import os
import random

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None
import numpy as np
import torch

from agent import DQNAgent
from env import DummyMinecraftEnv
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Configuration & Reproducibility
# ---------------------------------------------------------------------------
MODEL_DIR = "trained_models"
MISTAKE_MARGIN = 0.05
EPISODES = 10_000

# Replay buffer and optimisation parameters
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.995
N_STEPS = 3

# Exploration schedules
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5_000
TEMP_START = 1.0
TEMP_END = 0.1
TEMP_DECAY = 5_000

EVAL_INTERVAL = 1_000
EVAL_EPISODES = 10
TARGET_UPDATE_INTERVAL = 500

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
print(f"Using seed {SEED}")


def epsilon_by_episode(ep: int) -> float:
    """Exponential epsilon decay for more exploration early on."""
    return EPS_END + (EPS_START - EPS_END) * np.exp(-ep / EPS_DECAY)


def temperature_by_episode(ep: int) -> float:
    """Temperature schedule for Boltzmann exploration."""
    return max(TEMP_END, TEMP_START * np.exp(-ep / TEMP_DECAY))


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


def _load_latest_checkpoint(agent: DQNAgent) -> int:
    """Load the latest checkpoint if present and return the next episode."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoints = [
        f
        for f in os.listdir(MODEL_DIR)
        if f.startswith("model_ep") and f.endswith(".pth")
    ]
    if not checkpoints:
        return 1
    checkpoints.sort(key=lambda x: int(x[len("model_ep") : -len(".pth")]))
    latest = checkpoints[-1]
    agent.load(os.path.join(MODEL_DIR, latest))
    return int(latest[len("model_ep") : -len(".pth")]) + 1


def train() -> None:
    env = DummyMinecraftEnv()
    state_dim = len(env.reset())
    action_dim = 4

    agent = DQNAgent(state_dim, action_dim, gamma=GAMMA)
    start_ep = _load_latest_checkpoint(agent)
    buffer = ReplayBuffer(BUFFER_SIZE, gamma=GAMMA, n_step=N_STEPS)

    rewards: list[float] = []

    train_csv = "train_metrics.csv"
    eval_csv = "eval_metrics.csv"
    if not os.path.exists(train_csv):
        with open(train_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "episode",
                    "total_reward",
                    "avg_td_error",
                    "mistakes",
                    "epsilon",
                ]
            )
    if not os.path.exists(eval_csv):
        with open(eval_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "avg_reward", "win_rate", "avg_steps"]
            )

    for ep in range(start_ep, start_ep + EPISODES):
        state = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        mistakes = 0
        td_errors: list[float] = []

        epsilon = epsilon_by_episode(ep)
        temperature = temperature_by_episode(ep)

        while not done:
            q_values = agent.predict_q(state)
            action = agent.select_action(
                state, epsilon=epsilon, temperature=temperature, q_values=q_values
            )

            q_sa = float(q_values[action])
            max_q = float(np.max(q_values))
            if q_sa < max_q - MISTAKE_MARGIN:
                mistakes += 1

            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if len(buffer) >= BATCH_SIZE:
                batch, indices, weights = buffer.sample(BATCH_SIZE)
                td_errs = agent.train_step(batch, weights)
                buffer.update_priorities(indices, td_errs)
                td_errors.append(float(np.mean(np.abs(td_errs))))

        avg_td_error = float(np.mean(td_errors)) if td_errors else 0.0

        rewards.append(total_reward)
        with open(train_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, total_reward, avg_td_error, mistakes, epsilon]
            )

        if ep % EVAL_INTERVAL == 0:
            avg_reward_last = float(np.mean(rewards[-EVAL_INTERVAL:]))
            eval_avg_reward, eval_win_rate, eval_avg_steps, _ = evaluate(
                agent, EVAL_EPISODES
            )
            print(
                f"Episode {ep} - Avg Reward: {avg_reward_last:.3f} "
                f"Avg TD Error: {avg_td_error:.3f} "
                f"Mistakes: {mistakes} Epsilon: {epsilon:.3f} "
                f"Eval Avg Reward: {eval_avg_reward:.3f} Eval Win Rate: {eval_win_rate:.2f} "
                f"Eval Avg Steps: {eval_avg_steps:.2f}"
            )
            agent.save(os.path.join(MODEL_DIR, f"model_ep{ep}.pth"))
            with open(eval_csv, "a", newline="") as f:
                csv.writer(f).writerow(
                    [ep, eval_avg_reward, eval_win_rate, eval_avg_steps]
                )

        if ep % TARGET_UPDATE_INTERVAL == 0:
            agent.update_target()

    agent.save(os.path.join(MODEL_DIR, "model_final.pth"))

    if plt is not None:
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Rewards")
        plt.savefig("reward_plot.png")


if __name__ == "__main__":
    train()

