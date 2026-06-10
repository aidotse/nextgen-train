from typing import Any

from nextgen_train.src.core.components import ComponentFactory
from nextgen_train.src.core.factories import DIFactory
from nextgen_train.src.logging.utils import CLILoggerTrainerCallback


class ExpLoggerFactory(DIFactory):
    """Builds experiment loggers"""

    def get_dependencies(self) -> dict[str, Any]:
        return {}


class ExpLoggerTrainerCallbackFactory(DIFactory):
    """Builds Callbacks for experiment loggers"""

    def get_dependencies(self) -> dict[str, Any]:
        return {"logger": self.state.logger}


class CLILoggerTrainerCallbackFactory(ComponentFactory):
    """Builds Callbacks for CLI loggers"""

    def build(self) -> CLILoggerTrainerCallback:
        return CLILoggerTrainerCallback(**self.params)
