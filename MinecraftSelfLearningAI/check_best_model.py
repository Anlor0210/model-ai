import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from agent import DQNAgent
from env import DummyMinecraftEnv

MODEL_DIR = "trained_models"
RESULT_FILE = "best_model.txt"
EVAL_EPISODES = 10


def evaluate(agent: DQNAgent, episodes: int) -> float:
    """Evaluate the agent for a number of episodes without exploration."""
    env = DummyMinecraftEnv()
    rewards: list[float] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            q_values = agent.predict_q(state)
            action = int(np.argmax(q_values))
            state, reward, done = env.step(action)
            total += reward
        rewards.append(total)

    return float(np.mean(rewards)) if rewards else float("nan")


def main() -> None:
    if not os.path.isdir(MODEL_DIR):
        print(f"Directory '{MODEL_DIR}' does not exist.")
        return

    tmp_env = DummyMinecraftEnv()
    state_dim = len(tmp_env.reset())
    action_dim = 4
    del tmp_env

    agent = DQNAgent(state_dim, action_dim)

    best_avg = float("-inf")
    best_path = None

    for filename in os.listdir(MODEL_DIR):
        if not filename.endswith(".pth"):
            continue
        path = os.path.join(MODEL_DIR, filename)
        print(f"Evaluating {filename}...")
        try:
            agent.load(path)
            avg_reward = evaluate(agent, EVAL_EPISODES)
        except Exception as exc:
            print(f"Skipping {filename}: {exc}")
            continue

        if avg_reward > best_avg:
            best_avg = avg_reward
            best_path = path

    if best_path is not None:
        result = f"Model: {best_path} is the best model with avg reward: {best_avg:.2f}"
        print(result)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print("No valid model checkpoints found.")


if __name__ == "__main__":
    main()
