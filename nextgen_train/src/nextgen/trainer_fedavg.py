import gc
import math
import tempfile
import warnings
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from loguru import logger as cli_logger
from nextgen_framework.interfaces import BaseTrainer, TrainerArgs

from nextgen_train.src.core.callbacks import DDPStaticGraphCallback
from nextgen_train.src.core.state import ExperimentState
from nextgen_train.src.integrations.sentence_transformers.utils import safe_encode_context


class Trainer(BaseTrainer):
    """NextGen FedAvg Trainer.

    Created when a node (server or client) starts up, and then is passed to that node.
    This means: we can communicate state between the init method and the other methods,
    but these methods cannot communicate state with each other, which is why the model / training
    state has to be saved and loaded from disk.

    Logging strategy: =======================
    Logging happens on the per-node level. All runs share the same global run name, which can be
    either be given by the user or auto-generated if no name is provided. To differentiate runs
    with the same name in the backend, we also attach a tag to each run to identify the node.

    Example:
    - The master node's run will store all relevant data from `setup` and `reduce_models` methods.
    - Client 1 will have a single run spanning all of its training rounds, with a tag like `NODE: CLIENT_1`.
    - Client 2 will have a single run spanning all of its training rounds, with a tag like `NODE: CLIENT_2`.

    Component initialization: ================

    We group all components required for training into 2 groups:

    1. The static layer: These components are expensive to load, completely stateless regarding the training loop,
    and—most importantly, they have no dependencies on components that are part of the computation graph (see below)

    2. The ephemeral layer: These components form the active computational graph. The model itself is the first obvious
    element. Others may  hold VRAM, track step counts, etc. If a component depends on another component that lives in
    this layer, then it must also belong to this layer.

    The components in the ephemeral layer  must be systematically destroyed and restored across FL training runs to
    prevent memory leaks, orphaned processes, and state corruption across FL rounds. While this would work
    *functionally* using NVIDIA / AMD (eager mode), the consequences are harsher for Gaudi / in lazy mode:
    because Gaudi builds wrappers when e.g. traienr.train() is run, doing this twice in a row fails at runtime (it
    attempts to wrap the wrapper).

    Therefore, we rely on the notion of building and tearing down the computation graph on demand.
    """

    def __init__(self, cfg: TrainerArgs, node: Any, logger: None | Any):
        """Initializes all static assets and computation graph required for training."""
        super().__init__(cfg, node, logger)

        self.temp_dir = Path(tempfile.mkdtemp())
        self.expstate = ExperimentState(logger=logger)
        self.stretch_lr = self.cfg.stretch_lr
        if self.logger:
            self.logger.set_tag("NODE", self.source_str)  # Track the global step for logging purposes
            self.logger.set_global_step_offset(0)

        self._build_static_assets()
        self._build_hf_computation_graph()

        # Client-only setup
        if not self.node.is_master:
            # Shortcut for training without waiting to pull any global, or pushing anything
            if self.cfg.skip_fl:
                assert self.expstate.trainer is not None
                self.expstate.trainer.train()
                exit(0)

    def _build_static_assets(self) -> None:
        """Creates and updates the experiment state with Tier 1 (static) assets."""

        # "seed" model that we don't persist just to inspect inside this method. Gets an unrelated experiment state.
        model = instantiate(self.cfg.model, state=ExperimentState(), _convert_="none").build().to(self.cfg.device)

        # TOKENIZER
        # Get tokenizer either directly from the model if possible, otherwise from the config
        tokenizer = getattr(model, "tokenizer", None)
        cfg_tokenizer = getattr(self.cfg, "tokenizer", None)
        if tokenizer and cfg_tokenizer:
            cli_logger.bind(source=self.source_str).warning(
                "Found tokenizer in model, but also in config. Taking tokenizer from the config."
            )
            self.expstate.tokenizer = cfg_tokenizer
        elif tokenizer is None and cfg_tokenizer:
            self.expstate.tokenizer = (
                instantiate(self.cfg.tokenizer, state=self.expstate, _convert="none").build().to(self.cfg.device)
            )
        else:
            self.expstate.tokenizer = tokenizer

        # COLLATOR
        self.expstate.collator = instantiate(self.cfg.collator, state=self.expstate, _convert_="none").build()
        if (
            self.expstate.collator
            and model.model_max_length
            and model.model_max_length != self.expstate.collator.max_length
        ):
            cli_logger.bind(source=self.source_str).warning(
                f"Model max length ({model.model_max_length}) \
                doesnt match collator max length ({self.expstate.collator.max_length})."
            )

        # DATASETS
        data = instantiate(self.cfg.data, state=self.expstate, _convert_="none").build()
        self.expstate.train_dataset = data.train_dataset
        self.expstate.eval_dataset = data.eval_dataset

        model = None
        gc.collect()

    def _teardown_hf_computation_graph(self) -> None:
        """
        Systematically destroys all Tier 2 (Ephemeral) components. Assumes HF Trainer API.
        Work backwards through the dependency graph to dismantle components. You can see the dependency graph by
        looking at all the dependencies listed in e.g. src.integrations.sentence_transformers.factories.*:

            - Trainer depends on loss, model, callbacks, evaluators, etc..
            - loss depends on model, possibly also a guide model
            - evaluator may depend on model
            - ...

        NOTE: Many callbacks are actually highly stateful. (Example: EarlyStoppingCallback tracks `self.best_metric` and
        `self.patience_counter`. If these persist between FL rounds, a past training round's statistics may
        preemptively kill the training in a new round). Therefore: re-initialize.

        NOTE: we defensively also remove evaluators, even though most implementations pass the model via __call__ rather
        than __init__
        """

        cli_logger.bind(source=self.source_str).debug("Computation graph teardown ...")

        # trainer
        if getattr(self.expstate, "trainer", None) is not None:
            assert self.expstate.trainer is not None  # mypy
            # sever the core graph
            self.expstate.trainer.model = None
            if hasattr(self.expstate.trainer, "model_wrapped"):
                self.expstate.trainer.model_wrapped = None
            # sever the loss (break trainer -> loss -> model chain)
            if hasattr(self.expstate.trainer, "loss"):
                self.expstate.trainer.loss = None
            if hasattr(self.expstate.trainer, "loss_function"):
                self.expstate.trainer.loss_function = None
            # flush callback handler, defense again stateful callbacks / loggers
            if hasattr(self.expstate.trainer, "callback_handler"):
                self.expstate.trainer.callback_handler.callbacks = []
            # sever optimizer states
            self.expstate.trainer.optimizer = None
            self.expstate.trainer.lr_scheduler = None
            # shut down OS dataloader workers
            self.expstate.trainer.train_dataloader = None
            self.expstate.trainer.eval_dataloader = None
            self.expstate.trainer = None

        # experiment state
        if getattr(self, "expstate", None) is not None:
            self.expstate.callbacks = None
            self.expstate.guide = None
            self.expstate.loss = None
            self.expstate.evaluator = None
            self.expstate.model = None
            self.expstate.trainer = None

        # Sweep VRAM
        gc.collect()

    def _build_hf_computation_graph(self, weights_path: Path | None = None, version: str = "0.0.0") -> None:
        """
        Systematically reconstructs all Tier 2 (Ephemeral) components from scratch.
        All state is strictly maintained inside `self.expstate` to prevent duplicate pointer leaks.

        NOTE: On stretching or fast-forwarding the scheduler:

        Stretching the LR scheduler maintains a continuous learning rate schedule (e.g., a single linear decay)
        across all isolated Federated Learning rounds.

        By default, initializing a new trainer per round would restart the LR schedule, causing repeated warmups and
        effectively mimicking Cosine Annealing with Restarts (SGDR). To stretch a single schedule across the entire FL
        lifecycle, we natively build the HF scheduler using the total global steps and mathematically fast-forward it.

        Key Architectural Nuances:
        1. Fresh Optimizer (Zero Momentum): The optimizer MUST be re-instantiated from scratch every round. Server
        aggregation teleports the global model to a new coordinate in the loss landscape. Applying a client's old
        momentum vectors to this new coordinate would aggressively pull the model off the optimal path.

        2. The Adam Bias Correction Trap: During fast-forwarding, we explicitly step ONLY the scheduler, never the
        optimizer. Optimizers like Adam/AdamW track an internal step counter (t) to apply bias correction (1 - beta^t)
        to their initially empty momentum buffers. If we stepped the optimizer to match the global step, the bias
        correction would artificially vanish. Consequently, the genuinely empty momentum buffers at the start of the
        new round would cause the model's gradient updates to microscopically stall.
        """
        cli_logger.bind(source=self.source_str).debug("Building computation graph ...")

        # MODEL
        self.expstate.model = (
            instantiate(self.cfg.model, state=self.expstate, _convert_="none").build().to(self.cfg.device)
        )
        assert self.expstate.model is not None, "Model must exist in the computation graph!"

        # NOTE: depending on our strategy, we could load either all model weight tensors (safe),
        # or just the trainable ones (risky, but saves memory and bandwith). For this case, we need
        # to make sure that all clients as well as the master node initialize the model in the same way,
        # i.e. not with random weights! Specifically: make sure we use `from_pretrained` or loading via model string
        if weights_path:
            self.expstate.model.load_model_weights(weights_path)

        # LOSS
        # May require the guide model
        if "guide" in self.cfg:
            self.expstate.guide = (
                instantiate(self.cfg.guide, state=self.expstate, _convert_="none").build().to(self.cfg.device)
            )
        self.expstate.loss = instantiate(self.cfg.loss, state=self.expstate, _convert_="none").build()

        # EVALUATORS
        self.expstate.evaluator = instantiate(self.cfg.evaluator, state=self.expstate, _convert_="none").build()

        # CALLBACKS
        callbacks = []
        if "callbacks" in self.cfg:
            for _, cb_conf in self.cfg.callbacks.items():
                callbacks.append(instantiate(cb_conf, state=self.expstate, _convert_="none").build())
        callbacks.append(DDPStaticGraphCallback())
        self.expstate.callbacks = callbacks

        # DATALOADERS
        # NOTE: While datasets are stateless, their dataloader wrappers are not. In particular, consider the case when
        # each FL round trains less than an epoch: if we use the same starting seed, when the dataloaders are
        # re-initialized, they will return the same batches every FL round. That is, the full dataset is never explored.
        # Therefore: we use the version input number to incremenet the random seed *deterministically* to make sure
        # we get new batches in newer FL rounds, to make training continue "where it left off".
        global_round = int(version.split(".")[0])
        base_seed = self.cfg.trainer.args.get("seed", 42)
        self.cfg.trainer.args.data_seed = base_seed + global_round
        cli_logger.bind(source=self.source_str).debug(
            f"Set Trainer data_seed to {self.cfg.trainer.args.data_seed} for round {global_round}"
        )

        # TRAINER
        self.expstate.trainer = instantiate(self.cfg.trainer, state=self.expstate, _convert_="none").build()

        # Inject the trainer reference into our custom callback
        for cb in self.expstate.trainer.callback_handler.callbacks:
            if isinstance(cb, DDPStaticGraphCallback):
                cb.trainer = self.expstate.trainer

        # LR schedule for trainer
        if self.stretch_lr:
            # GLOBAL PROGRESS
            train_dataloader = self.expstate.trainer.get_train_dataloader()
            args = self.expstate.trainer.args

            # Calculate true optimization steps per FL round
            if args.max_steps > 0:
                steps_per_round = args.max_steps
            else:
                # Calculate based on dataloader, epochs, and gradient accumulation
                num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                # math.ceil protects against fractional epochs
                steps_per_round = math.ceil(num_update_steps_per_epoch * args.num_train_epochs)

            total_fl_rounds = self.node.max_iter
            current_global_step = global_round * steps_per_round
            total_global_steps = total_fl_rounds * steps_per_round

            # OPTIMIZER AND SCHEDULER
            # Optimizer creation is in-place for trainer, so we can access self.expstate.trainer.optimizer
            self.expstate.trainer.create_optimizer()
            # "Stretched" scheduler
            self.expstate.trainer.create_scheduler(
                num_training_steps=total_global_steps, optimizer=self.expstate.trainer.optimizer
            )

            # Fast-forward the scheduler to the current FL position
            if current_global_step > 0:
                cli_logger.bind(source=self.source_str).debug(
                    f"Fast-forwarding LR Scheduler by {current_global_step} steps..."
                )

                with warnings.catch_warnings():
                    # PyTorch throws a UserWarning if you step a scheduler before stepping the optimizer.
                    # We can safely ignore it since we are just doing math manipulation.
                    warnings.simplefilter("ignore", category=UserWarning)
                    for _ in range(current_global_step):
                        self.expstate.trainer.lr_scheduler.step()

    def setup(self) -> Path:
        """Runs on the server, creates the initial model"""

        cli_logger.bind(source=self.source_str).info("Running trainer setup")
        initial_model_path = self.temp_dir / "global_model_v0.pth"
        assert self.expstate.model is not None, "Model must exist in experiment state to load weights"
        self.expstate.model.save_model_weights(initial_model_path)
        cli_logger.bind(source=self.source_str).info(f"Initial model weights saved to {initial_model_path}")
        return Path(initial_model_path)

    def train(self, model_path: Path, version: str) -> Path:
        """
        Runs on the clients, runs a training session / round / epoch / etc.

        We implement a simple strategy here that resets all optimizer and trainer state
        between epochs / rounds. This is the default with e.g. FedAvg.
        """

        cli_logger.bind(source=self.source_str).info(f"Re-setting for ({version}) and model {model_path}...")
        self._teardown_hf_computation_graph()
        self._build_hf_computation_graph(model_path, version)
        assert self.expstate.trainer is not None, "Trainer must exist in experiment state for training!"
        assert self.expstate.model is not None, "Model must exist in experiment state for training!"
        cli_logger.bind(source=self.source_str).info("Starting training round")
        self.expstate.trainer.train()
        global_step_offset = "N/A"
        if self.logger:
            global_step_offset = self.logger.get_global_step_offset() + self.expstate.trainer.state.global_step
            self.logger.set_global_step_offset(global_step_offset)
        cli_logger.bind(source=self.source_str).info(
            f"Completed {self.expstate.trainer.state.global_step} steps. Total steps so far: {global_step_offset}"
        )
        cli_logger.bind(source=self.source_str).info("Training round completed.")

        # Save the newly trained local weights
        local_model_path = self.temp_dir / "local_model.pth"
        self.expstate.model.save_model_weights(local_model_path)
        cli_logger.bind(source=self.source_str).info(f"Local model weights saved to {local_model_path}")
        return local_model_path

    def reduce_models(self, model_paths: list[Path], version: str) -> Path:
        """
        Aggregates client models into a new global model using a memory-optimized FedAvg strategy.

        This method performs a *simple* average of model parameters on the CPU, specifically handling:
        1. **Trainable Parameters**: Aggregates weights/biases defined as trainable by the server.
        2. **Buffers**: Aggregates statistical buffers (e.g., BatchNorm `running_mean`, `running_var`) to prevent
            normalization drift.
        3. **Integer State**: Aggregates non-accumulatable state (e.g., `num_batches_tracked`) by taking the maximum
            value observed across clients.
        4. **Frozen Weights**: Merges aggregated updates back into the full server state, ensuring frozen parameters
            remain intact even if clients submit partial state dicts.

        Performance & Safety:
        - **RAM Optimized**: Loads one client model at a time to minimize memory spikes.
        - **Precision Guard**: Upcasts floating-point tensors to `float32` during accumulation to prevent overflow
            or precision loss (e.g., from FP16 inputs).
        - **Strict Consistency**: Enforces that all clients provide exactly the same set of keys for aggregation;
            raises `KeyError` on mismatch.

        Args:
            model_paths (list[Path]): List of file paths to the client model checkpoints (.pth).
            version (str): Version identifier for the resulting global model.

        Returns:
            Path: File path to the newly saved global model state dictionary.

        Raises:
            ValueError: If `model_paths` is empty.
            KeyError: If a client model is missing keys expected by the aggregation logic.
        """

        cli_logger.bind(source=self.source_str).info(f"Reducing {len(model_paths)} models.")

        if not model_paths:
            raise ValueError("Cannot perform aggregation: No model paths provided.")

        assert self.expstate.model, "Model must exist in experiment state"

        # IDENTIFY KEYS (Server Side)
        # We need to know what is trainable vs what is a buffer
        server_trainable = {n for n, p in self.expstate.model.model.named_parameters() if p.requires_grad}
        server_buffers = {n for n, b in self.expstate.model.model.named_buffers()}

        # LOAD FIRST CLIENT TO DEFINE THE AGGREGATION PROTOCOL
        first_client_path = model_paths[0]
        first_client_state = torch.load(first_client_path, map_location="cpu")
        client_keys = set(first_client_state.keys())

        # STRICT CHECK: Trainable Parameters
        # If a client is missing a weight we NEED to train, that's a critical failure.
        missing_trainable = server_trainable - client_keys
        if missing_trainable:
            raise KeyError(
                f"Client {first_client_path.name} is missing trainable keys expected by server: {missing_trainable}"
            )

        # LENIENT CHECK: Buffers
        # It is normal for some buffers (like position_ids) to be missing from disk (they are dropped from state_dict
        # when saved).
        # We only aggregate buffers that exist in BOTH the server list AND the client file.
        common_buffers = server_buffers.intersection(client_keys)

        # Optional: Warn if we are skipping buffers, just to be aware
        skipped_buffers = server_buffers - client_keys
        if skipped_buffers:
            cli_logger.bind(source=self.source_str).warning(
                f"Skipping {len(skipped_buffers)} non-persistent buffers (e.g. position_ids) missing from client: \
                {list(skipped_buffers)}"
            )

        # The final set of keys we will actually average
        keys_to_aggregate = server_trainable.union(common_buffers)

        # Initialize Accumulator
        aggregated_weights = {}
        original_dtypes = {}

        for k in keys_to_aggregate:
            data = first_client_state[k]
            original_dtypes[k] = data.dtype

            if torch.is_floating_point(data):
                aggregated_weights[k] = data.to(torch.float32)
            else:
                aggregated_weights[k] = data.clone()

        del first_client_state
        gc.collect()

        # ACCUMULATE
        for i, client_path in enumerate(model_paths[1:], start=1):
            cli_logger.bind(source=self.source_str).info(f"Processing client {i+1}/{len(model_paths)}")

            client_weights = torch.load(client_path, map_location="cpu")

            for key in aggregated_weights:
                if key not in client_weights:
                    raise KeyError(f"Consistency Mismatch: Client {client_path.name} missing '{key}'")

                # Logic for Integer Buffers (e.g. num_batches_tracked) -> Take MAX
                if not torch.is_floating_point(aggregated_weights[key]):
                    # Safe for 0 and > 0 dimensional scalar tensors
                    aggregated_weights[key] = torch.maximum(aggregated_weights[key], client_weights[key])
                    continue

                # Logic for Floats -> SUM
                # In-place add. client_weights[key] is auto-promoted to float32 if needed.
                aggregated_weights[key].add_(client_weights[key].to(aggregated_weights[key].dtype))

            del client_weights
            gc.collect()

        # AVERAGE AND RESTORE DTYPE
        n_models = len(model_paths)
        for key in aggregated_weights:
            # Only divide floating point values
            if torch.is_floating_point(aggregated_weights[key]):
                aggregated_weights[key].div_(n_models)
                # Restore original precision (e.g., float16)
                aggregated_weights[key] = aggregated_weights[key].to(original_dtypes[key])

            # Integer buffers are already correct (max value held)

        # MERGE WITH SERVER STATE
        # Start with Server's full state (preserves frozen weights and non-aggregated keys)
        full_state_to_save = self.expstate.model.model.state_dict()

        for k, v in aggregated_weights.items():
            full_state_to_save[k] = v

        # SAVE
        new_global_path = self.temp_dir / "global_model_aggregated.pth"
        torch.save(full_state_to_save, new_global_path)

        cli_logger.bind(source=self.source_str).info(f"Global model saved to {new_global_path}")

        return new_global_path

    def eval_model(self, model_path: Path, version: str) -> None:
        """
        Runs on the server. Evaluates a global model checkpoint using the predefined evaluator.
        """
        if not hasattr(self.expstate, "evaluator") or self.expstate.evaluator is None:
            cli_logger.bind(source=self.source_str).warning("No evaluator configured. Skipping evaluation.")
            return

        cli_logger.bind(source=self.source_str).info(
            f"Loading global model ({version}) from {model_path} for evaluation..."
        )

        # This uses the existing wrapper method to safely populate the PyTorch architecture
        assert self.expstate.model is not None, "Model must be defined for evaluation"
        self.expstate.model.load_model_weights(model_path)

        # Get the current global step for accurate X-axis plotting
        current_step = int(version.split(".")[0])

        # Execute the evaluator - we assume the evaluator takes in the raw model rather than our wrapper
        max_len = self.expstate.model.model_max_length
        assert max_len is not None, "Model max length must be defined for evaluation."
        with safe_encode_context(self.expstate.model.model, max_len, self.node.eval_batch_size) as safe_model:
            metrics = self.expstate.evaluator(safe_model)

        # Log the output dictionary to MLflow/your custom logger
        if self.logger and metrics:
            self.logger.log_metrics(metrics, step=current_step)
