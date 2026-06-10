import inspect
from abc import abstractmethod
from typing import Any

from hydra.utils import get_class

from .components import ComponentFactory


class DIFactory(ComponentFactory):
    """
    A generic dependency-injection factory that inspects the target class's signature and
    automatically injects matching dependencies from the provided context.
    """

    @abstractmethod
    def get_dependencies(self) -> dict[str, Any]:
        """
        Must be implemented by subclasses to define the pool of available
        runtime objects (e.g., {'model': state.model, 'tokenizer': ...})
        """

    def build(self):
        # 1. Get the implementation config
        #    We assume the YAML structure:
        #    _target_: ...DIFactory
        #    _target_class_: { _target_: ...RealClass, ... }
        target_path = self.params.pop("_target_class_", None)
        if not target_path:
            raise ValueError(f"'{type(self).__name__}' config missing '_target_class_' key.")
        target_cls = get_class(target_path)

        # 2. Get Dependencies
        candidates = self.get_dependencies()

        # 3. Introspection logic
        final_kwargs = self.params.copy()

        # 4. Inject Dependencies
        sig = inspect.signature(target_cls.__init__)
        for param_name in sig.parameters:
            if param_name in candidates:
                val = candidates[param_name]
                if val is not None:
                    final_kwargs[param_name] = val

        # 5. Direct Instantiation
        return target_cls(**final_kwargs)
