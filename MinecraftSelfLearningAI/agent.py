"""Agent and neural network definitions for Minecraft DQN experiments.

This module provides a slightly deeper network than the original
implementation and incorporates a number of stabilisation techniques:

* Four fully-connected layers with Batch Normalisation for richer feature
  extraction.  The network remains small enough to train quickly on CPU
  machines such as Windows laptops.
* Optional LSTM layer for environments that require sequence modelling.
* Boltzmann exploration, gradient clipping and a learning-rate scheduler.
* Double DQN targets for reduced Q-value overestimation.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """Feed-forward network with optional LSTM layer."""

    def __init__(self, input_dim: int, output_dim: int, use_lstm: bool = False) -> None:
        super().__init__()
        self.use_lstm = use_lstm

        hidden = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        if use_lstm:
            # LSTM operates on the feature dimension; batch_first makes it easy
            # to work with single-step batches.
            self.lstm = nn.LSTM(64, 64, batch_first=True)

        self.head = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.net(x)
        if self.use_lstm:
            x = x.unsqueeze(1)
            x, _ = self.lstm(x)
            x = x.squeeze(1)
        return self.head(x)


class DQNAgent:
    """Deep Q-Network based agent with Double DQN updates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_lstm: bool = False,
        grad_clip: float = 1.0,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma
        self.grad_clip = grad_clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim, use_lstm=use_lstm).to(self.device)
        self.target_net = DQN(state_dim, action_dim, use_lstm=use_lstm).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10_000, gamma=0.9)
        # Huber loss with 'none' reduction so that importance weights can be
        # applied when using prioritized replay.
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(
        self,
        state: Tuple[int, ...],
        epsilon: float = 0.0,
        temperature: float = 1.0,
        q_values: Optional[np.ndarray] = None,
    ) -> int:
        """Select an action using Boltzmann exploration.

        Parameters
        ----------
        state : Tuple[int, ...]
            Environment state.
        epsilon : float, optional
            Probability of taking a purely random action.
        temperature : float, optional
            Temperature parameter for the softmax exploration.  Higher values
            yield more uniform sampling; lower values approach greedy
            selection.
        q_values : np.ndarray, optional
            Pre-computed Q-values for *state* to avoid an extra forward pass.
        """

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        if q_values is None:
            q_values = self.predict_q(state)
        q_tensor = torch.tensor(q_values / max(temperature, 1e-6), dtype=torch.float32)
        probs = torch.softmax(q_tensor, dim=0).cpu().numpy()
        return int(np.random.choice(self.action_dim, p=probs))

    def predict_q(self, state: Tuple[int, ...]) -> np.ndarray:
        """Return Q-values for a given state as a NumPy array."""

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t).cpu().numpy()[0]
        return q_values

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def train_step(self, batch, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Update the policy network using a batch of experience.

        Parameters
        ----------
        batch : tuple
            (states, actions, rewards, next_states, dones, n_steps)
        weights : np.ndarray, optional
            Importance-sampling weights for prioritized replay.

        Returns
        -------
        np.ndarray
            TD errors for each element in the batch, useful for updating
            priorities in a replay buffer.
        """

        states, actions, rewards, next_states, dones, n_steps = batch

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        n_steps_t = torch.tensor(n_steps, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1)
            next_q_values = self.target_net(next_states_t).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            targets = rewards_t + (self.gamma ** n_steps_t) * next_q_values * (1 - dones_t)

        td_errors = targets - q_values
        losses = self.loss_fn(q_values, targets)
        if weights is not None:
            weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
            loss = (losses * weights_t).mean()
        else:
            loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return td_errors.detach().cpu().numpy()

    def update_target(self) -> None:
        """Synchronize target network with policy network."""

        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.update_target()

