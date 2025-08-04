import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    """Simple feed-forward network used to approximate Q-values."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


class DQNAgent:
    """Deep Q-Network based agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ) -> None:
        self.action_dim = action_dim
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state: Tuple[int, ...], epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch) -> None:
        """Update the policy network using a batch of experience."""

        states, actions, rewards, next_states, dones = batch

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0]
            targets = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self) -> None:
        """Synchronize target network with policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

