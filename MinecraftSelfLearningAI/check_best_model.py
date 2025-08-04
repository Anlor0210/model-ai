"""Evaluate saved models and pick the best one."""

import csv
import os
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from agent import DQNAgent
from env import DummyMinecraftEnv

MODEL_DIR = "trained_models"
EVAL_CSV = "evaluation_metrics.csv"
RESULT_FILE = "best_model.txt"
EVAL_EPISODES = 10


def evaluate(agent: DQNAgent, episodes: int) -> tuple[float, float, float]:
    """Run evaluation episodes without exploration.

    Puts the policy network into evaluation mode so that BatchNorm layers use
    running statistics rather than per-batch statistics.
    """

    agent.policy_net.eval()
    env = DummyMinecraftEnv()
    rewards: list[float] = []
    steps_list: list[int] = []
    wins = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            q_values = agent.predict_q(state)
            action = int(np.argmax(q_values))
            state, reward, done = env.step(action)
            total += reward
            steps += 1
        rewards.append(total)
        steps_list.append(steps)
        if reward > 0:
            wins += 1

    avg_reward = float(np.mean(rewards)) if rewards else float("nan")
    win_rate = wins / episodes if episodes else float("nan")
    avg_steps = float(np.mean(steps_list)) if steps_list else float("nan")
    agent.policy_net.train()
    return avg_reward, win_rate, avg_steps


def main() -> None:
    if not os.path.isdir(MODEL_DIR):
        print(f"Directory '{MODEL_DIR}' does not exist.")
        return

    tmp_env = DummyMinecraftEnv()
    state_dim = len(tmp_env.reset())
    action_dim = 4
    del tmp_env

    agent = DQNAgent(state_dim, action_dim)

    if not os.path.exists(EVAL_CSV):
        with open(EVAL_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["model", "avg_reward", "win_rate", "avg_steps"])

    best_avg = float("-inf")
    best_path = None

    for filename in sorted(os.listdir(MODEL_DIR)):
        if not filename.endswith(".pth"):
            continue
        path = os.path.join(MODEL_DIR, filename)
        print(f"Evaluating {filename}...")
        try:
            agent.load(path)
            avg_reward, win_rate, avg_steps = evaluate(agent, EVAL_EPISODES)
        except Exception as exc:
            print(f"Skipping {filename}: {exc}")
            continue

        print(
            f"Model {filename}: Avg Reward {avg_reward:.2f}, Win Rate {win_rate:.2f}, Avg Steps {avg_steps:.2f}"
        )

        with open(EVAL_CSV, "a", newline="") as f:
            csv.writer(f).writerow([filename, avg_reward, win_rate, avg_steps])

        if avg_reward > best_avg:
            best_avg = avg_reward
            best_path = path

    if best_path is not None:
        os.makedirs("best_model", exist_ok=True)
        dest_path = os.path.join("best_model", "best_model.pth")
        shutil.copy(best_path, dest_path)
        result = f"Model: {best_path} is the best model with avg reward: {best_avg:.2f}"
        print(result)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print("No valid model checkpoints found.")


if __name__ == "__main__":
    main()

