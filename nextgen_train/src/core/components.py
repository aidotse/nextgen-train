from abc import ABC, abstractmethod
from typing import Any

from nextgen_train.src.core.state import ExperimentState


class ComponentFactory(ABC):
    """An abstract component, takes a configuration object (dict) as well as the current experiment state."""

    def __init__(self, state: ExperimentState, **kwargs):
        self.state = state
        self.params = kwargs

    @abstractmethod
    def build(self) -> Any:
        """Constructs and returns the actual object (Model, Loss, etc.)"""
