import copy

import torch
from loguru import logger as cli_logger
from optimum.habana import GaudiTrainer
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from torch import nn
from transformers.trainer_callback import TrainerControl

from nextgen_train.src.integrations.sentence_transformers.utils import safe_encode_context


class STGaudiTrainer(GaudiTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        eval_dataset,
        data_collator,
        callbacks,
        loss,
        evaluator: SentenceEvaluator = None,
        *args,
        **kwargs,
    ):
        """
        A GaudiTrainer for Sentence Transformers: allows injection of a loss object and evaluator. This is unlike
        HF where e.g. the loss is owned directly by the model. Suitable for cpu, GPU, HPU, and multi-GPU/HPU training.
        """
        super().__init__(
            *args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
            **kwargs,
        )
        self.loss_func = loss
        self.evaluator = evaluator
        # Unwrap to get the raw model structure
        self.raw_model = self.accelerator.unwrap_model(self.model)
        # shallow Copy to isolate evaluation logic from training logic
        self.eval_model = copy.copy(self.raw_model)
        # Expect models to have max_length for e.g. tokenization and padding
        self.max_len = getattr(model, "max_seq_length", None)
        if not self.max_len:
            raise ValueError("Model does not have a `max_seq_length`")
        self._checked_data = False
        self._checked_grads = False

    def compute_loss(self, model: nn.Module, inputs, return_outputs=False, **kwargs):
        """
        Generic compute_loss that handles:
        1. Stateful Losses (e.g., GIST, Contrastive) -> Performs DDP Hot-Swap
        2. Stateless Losses (e.g., CrossEntropy) -> Standard execution

        The Trainer passes us the potentially DDP-wrapped model for the current step directly in this function.
        This works as follows: `model` gets passed to the input -> we send it to GaudiTrainer, which saves `self.model`,
        the Trainer then picks this up later and wraps it in DDP if needed. The result is then passed to this method.
        """

        # Problem: stateful losses have a copy of the model inside them, which is NOT the same reference as the
        # Trainer's model. This causes issues with DDP because the Trainer wraps the model in DDP after we pass it in.
        # Solution: "swap" the models to make sure that we use the potentially DDP-wrapped one.
        # We only try to swap the model if the loss function actually HAS a model.
        # This prevents crashes when using simple losses like CrossEntropy.
        is_stateful_loss = hasattr(self.loss_func, "model")
        original_model_ref = None
        if is_stateful_loss:
            original_model_ref = self.loss_func.model
            self.loss_func.model = model

        try:
            # Hijack the following (stateless) utility function in SentenceTransformerTrainer. Although it expects
            # 'self' (a SentenceTransformerTrainer) as a first argument, send 'None' here which crash (desired) if ever
            # their implementation changes to a stateful one.
            features, labels = SentenceTransformerTrainer.collect_features(None, inputs)
            loss = self.loss_func(features, labels)

        finally:
            if is_stateful_loss and original_model_ref is not None:
                self.loss_func.model = original_model_ref

        return (loss, None) if return_outputs else loss

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval") -> dict[str, float]:
        """
        Overridden to run a given SentenceEvaluator (passed in __init__) if present.
        This allows 'eval_steps' and 'load_best_model_at_end' to work naturally.

        Note: Implications for DDP. We don't mirror any sharding & gathering, i.e. ALL GPUs will run this evaluation.
        Due to syncing issues this behavior is desired vs. "gating" all but rank 0 here.
        """
        if self.evaluator is not None:
            # For DDP, only rank 0 gets to write to disk to prevent FileNotFoundError and log corruption.
            eval_output_dir = self.args.output_dir if self.is_world_process_zero() else None

            # Execute evaluation within our hardened context manager
            # This handles static tokenization, manual module iteration, and bfloat16-to-float32 casting
            # NOTE: if getting errors here try setting batch_size=1, this enables eager mode apparently
            batch_size = self.args.per_device_eval_batch_size
            if not self.max_len:
                raise ValueError("Model does not have a `max_seq_length`")
            with safe_encode_context(self.eval_model, self.max_len, batch_size) as patched_model:
                metrics = self.evaluator(
                    patched_model, output_path=eval_output_dir, epoch=self.state.epoch, steps=self.state.global_step
                )

            # Standard HF Trainer expects validation metrics to start with "eval_"
            # We map "ir-eval_mrr" -> "eval_ir-eval_mrr"
            if metric_key_prefix:
                metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

            self.log(metrics)
            self.control: TrainerControl = self.callback_handler.on_evaluate(
                self.args, self.state, self.control, metrics
            )
            return metrics

        cli_logger.bind(source=type(self).__name__).info(
            "No SentenceEvaluator passed, falling back on default evaluation (eval_loss)"
        )
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        SentenceTransformer models do not accept unpacked arguments: model(**inputs), only a single dictionary
        argument. This needs patching.

        Required for super().evaluate() to work.

        Robustly handles both Supervised (2 args) and Unsupervised (1 arg) losses.
        """
        inputs = self._prepare_inputs(inputs)

        # see `compute_loss`
        features, labels = SentenceTransformerTrainer.collect_features(None, inputs)

        loss = None
        if self.loss_func is not None:
            try:
                # 'features' is already the List[Dict] structure the loss expects
                # 'labels' is the Tensor (or None)
                loss = self.loss_func(features, labels)
            except Exception:
                # Fallback for semantic mismatches (e.g. Unsupervised loss + Labels,
                # or Single Sentence + Pair Loss)
                loss = torch.tensor(0.0, device=self.args.device)

        return (loss, None, labels)

    def training_step(self, model, inputs, num_items_in_batch):
        """
        For DDP, we need to force  the loss function to use the DDP-wrapped model!
        Otherwise the DDP hooks are not made available to the loss, i.e. as far as Pytorch is concerned, the DDP
        model was never used for the forward pass!

        Also, check for silent, but fatal bugs in DDP:

        1. The "clone" (data duplication) bug - the DataLoader should have a DistributedSampler that prevents the exact
            same data being loaded on all HPUs. Expect to see: *different* tokens across ranks

        2. The DDP Bypass bug - if we e.g. mess with the DDP wrappings that the Trainer has access to, we can
            inadvertently destroy the "network" hooks that force the HPUs to pause and average gradients across the
            network. In this case, each HPU will appear busy and train normally, but they will each train their own
            model that never gets averaged back with the rest. Expect to see: the exact same gradients across ranks
            (after they are gathered and averaged)
        """
        local_rank = self.args.local_rank if self.args.local_rank > -1 else 0

        # Test for data duplication bug : all HPUs should have different data
        if self.state.global_step == 1 and not self._checked_data:
            cli_logger.bind(source="DDP_CHECK").info(f"[Rank {local_rank}] Available Input Keys: {list(inputs.keys())}")
            first_tokens = None
            check_first_tokens = 20
            # Dynamically hunt for the 'input_ids' array
            for key, val in inputs.items():
                # Case A: Nested dictionary (e.g., inputs["sentence_0"]["input_ids"])
                if isinstance(val, dict) and "input_ids" in val:
                    first_tokens = val["input_ids"][0][:check_first_tokens]
                    break
                # Case B: Flattened keys (e.g., inputs["text1_input_ids"])
                elif isinstance(key, str) and "input_ids" in key:
                    first_tokens = val[0][:check_first_tokens]
                    break
                # Case C: List of features (e.g., inputs["features"][0]["input_ids"])
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and "input_ids" in val[0]:
                    first_tokens = val[0]["input_ids"][0][:check_first_tokens]
                    break
                else:
                    cli_logger.bind(source="DDP_CHECK").error(
                        f"[Rank {local_rank}] CRITICAL: Could not locate 'input_ids' inside the batch!"
                    )

            cli_logger.bind(source="DDP_CHECK").info(f"[Rank {local_rank}] Step 1 Data Tokens: {first_tokens}")
            self._checked_data = True

        # Force the loss function to use the DDP-wrapped model!
        # Otherwise the DDP hooks are not made available to the loss, i.e. as far as Pytorch is concerned, the DDP
        # model was never used for the forward pass!
        # Below: in DDP, self.loss_func.model is e.g. a SentenceTransformer, and model is a DDP wrapper
        if hasattr(self.loss_func, "model") and self.loss_func.model is not model:
            cli_logger.bind(source="DDP_CHECK").info(f"[Rank {local_rank}] Patching Loss Function to use DDP Model!")
            self.loss_func.model = model

        loss = super().training_step(model, inputs)

        # Test for DDP bypass: after the forward and backward pass, gradients should be gathered and averaged, and
        # therefore gradients for a specific weight layer on every HPU should be mathematically identical.
        if self.state.global_step == 1 and not self._checked_grads:
            # Grab a dense layer at the end of the model, where we can expect non zero gradients.
            # Search backwards from the end of the model to find the first active gradient
            check_num_gradients = 10
            for param in reversed(list(self.model.parameters())):
                if param.grad is not None:
                    # We found a layer that was actually updated!
                    sample_grad = param.grad.flatten()[:check_num_gradients]
                    grad_sum = param.grad.abs().sum().item()

                    cli_logger.bind(source="DDP_CHECK").info(
                        f"[Rank {local_rank}] Step 1 Gradients (Sum: {grad_sum:.4f}): {sample_grad}"
                    )
                    self._checked_grads = True
                    break

            # Failsafe if the entire model is frozen or broken
            if not self._checked_grads:
                cli_logger.bind(source="DDP_CHECK").error(
                    f"[Rank {local_rank}] CRITICAL: No gradients found anywhere in the model!"
                )

        return loss
