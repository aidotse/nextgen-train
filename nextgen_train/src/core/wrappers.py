from abc import ABC, abstractmethod
from pathlib import Path

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn


class BaseWrapper(nn.Module, ABC):
    """
    Abstract Base Class for Model Wrappers.

    Enforces the interface required by the Federated Learning Trainer/Server:
    1. access to the inner model via `.model` (for aggregation)
    2. standardized IO methods (`load_model_weights`, `save_model_weights`)
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """
        Must return the underlying PyTorch model object.
        Used by the server to access .named_parameters() and .named_buffers().
        """

    @property
    @abstractmethod
    def model_max_length(self) -> None | int:
        """The model's built-in max length, if available, for e.g. sequence padding"""

    @property
    @abstractmethod
    def tokenizer(self) -> None | nn.Module:
        """The model's internal tockenizer, if available."""

    def to(self, device: str) -> Self:
        """Model placement"""
        self.model.to(device)
        return self

    def load_model_weights(self, path: Path, device: str = "cpu") -> Self:
        """Generic PyTorch loader. Loads to CPU first to save VRAM."""
        # We assume the file contains a state_dict
        state_dict = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        # Optional: ensure it moves back to the right device if needed
        if hasattr(self, "device"):
            self.model.to(self.device)
        return self

    def save_model_weights(self, path: Path) -> None:
        """Generic PyTorch saver."""
        torch.save(self.model.state_dict(), path)

    def forward(self, *args, **kwargs):
        """Passes inputs through to the underlying model."""
        return self.model(*args, **kwargs)

    """
    def __getattr__(self, name: str):

        Forward missing attributes (like 'max_seq_length') to the inner model.
        This makes the wrapper 'transparent' to external libraries.

        try:
            # We use super().__getattribute__ to avoid infinite recursion
            # if 'self.model' is not yet defined.
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    """
