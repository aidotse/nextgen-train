from loguru import logger
from transformers import TrainerCallback


class DDPStaticGraphCallback(TrainerCallback):
    """
    Forces static_graph on the DDP wrapper to survive e.g. MNRL checkpointing.
    Safely disables itself if running on a single device.
    """

    def __init__(self):
        self.patched = False
        self.trainer = None

    def on_step_begin(self, args, state, control, **kwargs):
        # Skip if already patched (or permanently disabled)
        # Check if we are even running Distributed Data Parallel!
        # If world_size is 1 (or less), DDP isn't active. Exit silently.
        if self.patched or args.world_size <= 1:
            self.patched = True  # Permanently bypass for the rest of the run
            return

        # Helper function to apply the fix
        def _apply_static(m):
            if hasattr(m, "_set_static_graph"):
                m._set_static_graph()
                logger.bind(source=type(self).__name__).info(f"Forced static_graph on {type(m)}")
                return True
            return False

        # Grab the DDP wrapper directly from the injected trainer
        if self.trainer is not None and hasattr(self.trainer, "model_wrapped"):
            active_model = self.trainer.model_wrapped
        else:
            # Fallback to the payload if the trainer wasn't injected
            active_model = kwargs.get("model")

        if active_model is not None:
            # Check the top-level wrapper first
            if _apply_static(active_model):
                self.patched = True
                return

            # Otherwise, hunt through the children
            for _name, child in active_model.named_modules():
                if _apply_static(child):
                    self.patched = True
                    return

        logger.bind(source=type(self).__name__).warning("Running DDP (world_size > 1) but could not find wrapper!")
        self.patched = True
