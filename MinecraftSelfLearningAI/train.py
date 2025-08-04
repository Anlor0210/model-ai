from env import DummyMinecraftEnv
from agent import QLearningAgent
import matplotlib.pyplot as plt

env = DummyMinecraftEnv()
agent = QLearningAgent(actions=[0, 1, 2, 3])

episodes = 5000
reward_log = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break

    reward_log.append(total_reward)
    if ep % 500 == 0:
        print(f"Episode {ep} - Reward: {total_reward:.2f}")

# Plot rewards
plt.plot(reward_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("AI Learning Progress")
plt.savefig("reward_plot.png")
plt.show()