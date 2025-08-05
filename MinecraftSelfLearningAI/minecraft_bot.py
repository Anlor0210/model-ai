"""Minecraft inference bot using a pre-trained DQN model.

This module provides a thin environment wrapper that captures the game screen,
parses a small inventory region and issues keyboard / mouse commands based on
the model's output.  It is intentionally lightweight and avoids game specific
libraries such as Mineflayer.

The bot expects a Minecraft Java Edition 1.12.2 instance to be focused on the
local machine and connected to ``localhost:60025``.  All interaction happens
via OS-level events (simulated key presses and relative mouse movement).

Example
-------
Run the bot from the project root::

    python MinecraftSelfLearningAI/minecraft_bot.py

This will load ``trained_models/model_ep16000.pth`` and continuously query the
model for actions.  The environment state is composed of a coarse visual
encoding of the screen together with a short action history, allowing the
network to decide purely from learned behaviour.

Note
----
The implementation focuses on the infrastructure required for inference.  It
does not attempt to craft specific behaviour or encode rules for crafting.
The quality of the actions therefore depends entirely on the supplied model.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List

import numpy as np
import pyautogui
import torch
from mss import mss
from PIL import Image

from agent import DQN

# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class MinecraftEnvWrapper:
    """Capture screen, inventory bar and maintain action history.

    The wrapper grabs a predefined region of the screen, downsamples it and
    flattens pixel intensities to create a compact observation vector.  In
    addition, a narrow band at the bottom of the window is captured to provide a
    coarse representation of the hotbar / inventory.  A deque stores the IDs of
    the most recent actions which are appended to the observation so that the
    agent can reason about short term context.
    """

    def __init__(self, region: Dict[str, int] | None = None, history: int = 4):
        self.sct = mss()
        # Default to 854x480 window which matches the default Minecraft size
        self.region = region or {"top": 0, "left": 0, "width": 854, "height": 480}

        # A small region at the bottom is used for inventory / hotbar parsing
        self.inv_region = {
            "top": self.region["top"] + self.region["height"] - 60,
            "left": self.region["left"],
            "width": self.region["width"],
            "height": 60,
        }

        self.history: Deque[int] = deque([0] * history, maxlen=history)
        self.history_len = history

        # Compute state dimension by sampling once
        sample = self._capture()
        inv_sample = self._capture_inventory()
        self.vision_dim = sample.size
        self.inventory_dim = inv_sample.size

    def reset(self) -> np.ndarray:
        self.history.clear()
        self.history.extend([0] * self.history_len)
        return self.get_state()

    def _capture(self) -> np.ndarray:
        raw = self.sct.grab(self.region)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img = img.resize((32, 32))  # Coarse representation to keep state small
        arr = np.asarray(img, dtype=np.float32).ravel() / 255.0
        return arr

    def _capture_inventory(self) -> np.ndarray:
        raw = self.sct.grab(self.inv_region)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img = img.resize((32, 8))
        arr = np.asarray(img, dtype=np.float32).ravel() / 255.0
        return arr

    def get_state(self) -> np.ndarray:
        vision = self._capture()
        inventory = self._capture_inventory()
        hist = np.array(self.history, dtype=np.float32)
        return np.concatenate([vision, inventory, hist])

    @property
    def state_dim(self) -> int:
        return self.vision_dim + self.inventory_dim + self.history_len

    def step(self, action: int) -> np.ndarray:
        self.history.append(action)
        time.sleep(0.05)  # allow the game to react
        return self.get_state()


# ---------------------------------------------------------------------------
# Action handler
# ---------------------------------------------------------------------------

class ActionHandler:
    """Map discrete actions to keyboard/mouse events."""

    def __init__(self) -> None:
        # The pre-trained checkpoint was trained with four discrete actions.
        # Only the basic movement keys are retained here so that the network's
        # output dimension matches the checkpoint's ``head.weight`` shape
        # ``(4, 64)``.
        self.actions: List[callable] = [
            lambda: pyautogui.press("w"),
            lambda: pyautogui.press("a"),
            lambda: pyautogui.press("s"),
            lambda: pyautogui.press("d"),
        ]

    def __len__(self) -> int:
        return len(self.actions)

    def perform(self, action_id: int) -> None:
        if 0 <= action_id < len(self.actions):
            self.actions[action_id]()


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def main() -> None:
    env = MinecraftEnvWrapper()
    actions = ActionHandler()

    # Instantiate network with the dimensions used during training.  The first
    # layer now expects a 6-element state vector and the output layer produces
    # four Q-values corresponding to the retained actions.
    model = DQN(6, 4)
    state_dict = torch.load(
        "trained_models/model_ep16000.pth", map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    state = env.reset()
    while True:
        with torch.no_grad():
            # Only the first six elements are fed to the network so that the
            # input size matches the checkpoint's architecture.
            q_values = model(torch.tensor(state[:6], dtype=torch.float32)).numpy()
        action = int(np.argmax(q_values))
        actions.perform(action)
        state = env.step(action)


if __name__ == "__main__":
    main()
