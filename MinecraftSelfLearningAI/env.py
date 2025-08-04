import random

class DummyMinecraftEnv:
    def __init__(self, size=7):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [self.size//2, self.size//2]
        self.food = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        self.zombie = [random.randint(0, self.size-1), random.randint(0, self.size-1)]
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        return tuple(self.agent_pos + self.food + self.zombie)

    def step(self, action):
        # 0: up, 1: down, 2: left, 3: right
        if action == 0 and self.agent_pos[0] > 0: self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size-1: self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0: self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size-1: self.agent_pos[1] += 1

        self.steps += 1
        done = False
        reward = -0.01  # default small penalty

        if self.agent_pos == self.zombie:
            reward = -1
            done = True
        elif self.agent_pos == self.food:
            reward = 1
            done = True
        elif self.steps >= 50:
            done = True

        return self._get_state(), reward, done