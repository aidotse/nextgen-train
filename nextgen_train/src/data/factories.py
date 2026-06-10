from typing import Any

from nextgen_train.src.core.factories import DIFactory


class PharmaQADataFactory(DIFactory):
    """Builds datasets for PharmaQA use cases."""

    def get_dependencies(self) -> dict[str, Any]:
        """Specify dependencies for pharma datasets"""

        return {}
