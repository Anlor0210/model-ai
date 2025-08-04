"""Replay buffer with optional prioritized experience and N-step returns."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-size buffer supporting Prioritized Experience Replay and N-step returns."""

    def __init__(
        self,
        capacity: int,
        gamma: float,
        n_step: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.gamma = gamma
        self.n_step = n_step
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        # Main storage
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]] = []
        self.priorities: List[float] = []
        self.pos = 0

        # Temporary buffer for N-step returns
        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=n_step
        )

    # ------------------------------------------------------------------
    def _add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool, int]) -> None:
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def _get_n_step_info(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool, int]:
        R = 0.0
        next_state, done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, reward, _, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * reward
        state, action = self.n_step_buffer[0][:2]
        return state, action, R, next_state, done, len(self.n_step_buffer)

    def push(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
    ) -> None:
        transition = (np.array(state), action, reward, np.array(next_state), done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        state, action, R, next_state, done, steps = self._get_n_step_info()
        self._add((state, action, R, next_state, done, steps))

        # If episode ended, flush the remaining buffer
        if done:
            while self.n_step_buffer:
                self.n_step_buffer.popleft()
                if self.n_step_buffer:
                    state, action, R, next_state, done, steps = self._get_n_step_info()
                    self._add((state, action, R, next_state, done, steps))
        else:
            self.n_step_buffer.popleft()

    # ------------------------------------------------------------------
    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            raise ValueError("The replay buffer is empty")

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, n_steps = map(np.array, zip(*samples))

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (states, actions, rewards, next_states, dones, n_steps), indices, weights

    def update_priorities(self, indices, priorities) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(abs(prio) + 1e-6)

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return len(self.buffer)

