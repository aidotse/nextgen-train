from dataclasses import dataclass
from typing import Any

from nextgen_train.src.core.wrappers import BaseWrapper


@dataclass
class ExperimentState:
    """
    Central context object that holds the runtime artifacts of the experiment. Passed to all component factories.
    Note, this is different from a config: we don't use this to save all experiment hyperparameters. Rather, this is
    designed as a central registry for the artifacts created at runtime to solve dependency injection problems when
    e.g. switching low-level implementations (e.g. SentenceTransformers -> Transformers, etc.)
    """

    # Whatever model
    model: BaseWrapper | None = None
    # A guide model for e.g. guided in-batch sampling and other knowledge distillation approaches
    guide: BaseWrapper | None = None
    # Training dataset
    train_dataset: Any | None = None
    # Evaluation datset
    eval_dataset: Any | None = None
    # Data collator
    collator: Any | None = None
    # Tokenizer
    tokenizer: Any | None = None
    # Loss object, fed to Trainer
    loss: Any | None = None
    # Evaluator inputs for Trainer
    evaluator: Any | None = None
    # Experiment logger
    logger: Any | None = None
    # Callbacks
    callbacks: list[Any] | None = None
    # Trainer
    trainer: Any | None = None
