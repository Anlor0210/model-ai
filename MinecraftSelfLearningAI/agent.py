import random
import numpy as np

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return int(np.argmax(self.get_qs(state)))

    def learn(self, state, action, reward, next_state, done):
        current_q = self.get_qs(state)[action]
        max_next_q = np.max(self.get_qs(next_state)) if not done else 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q