from collections.abc import Callable
from typing import Any

from datasets import Dataset, load_dataset
from hydra.utils import instantiate
from loguru import logger as cli_logger
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from transformers import TrainerCallback


class STSBEvaluator(EmbeddingSimilarityEvaluator):
    """
    Subclass that automatically fetches the STS Benchmark dataset
    to act as an Anisotropy / Representation Collapse tripwire.
    """

    def __init__(self, eval_dataset: Dataset, name: str = "sts-b-dev", **kwargs):
        cli_logger.bind(source=type(self).__name__).info("Downloading STS Benchmark validation set...")
        stsb_dataset = load_dataset("sentence-transformers/stsb", split="validation")

        # Wrap the PyArrow columns in standard Python list() calls
        # This prevents the numpy.int64 type-clash on AMD/NVIDIA while remaining 100% safe for Gaudi.
        super().__init__(
            sentences1=list(stsb_dataset["sentence1"]),
            sentences2=list(stsb_dataset["sentence2"]),
            scores=list(stsb_dataset["score"]),
            main_similarity="cosine",
            name=name,
            **kwargs,
        )


class IREvaluator(InformationRetrievalEvaluator):
    """An Information Retrieval Evaluator from Sentence Transformers, handling dataset preparation."""

    def __init__(self, eval_dataset: Dataset, name: str = "IREvaluator", **kwargs):
        cli_logger.bind(source=type(self).__name__).info("Creating evaluation set for Information Retrieval...")

        if len(eval_dataset) == 0:
            raise ValueError("Evaluation dataset is empty.")

        # Detect the dataset schema
        columns = eval_dataset.column_names if hasattr(eval_dataset, "column_names") else list(eval_dataset[0].keys())

        is_in_batch = "anchor" in columns and "positive" in columns
        is_hard_negative = is_in_batch and "negative" in columns

        if is_hard_negative:
            cli_logger.bind(source=type(self).__name__).info(f"Detected hard negative schema: {columns}")
        elif is_in_batch:
            cli_logger.bind(source=type(self).__name__).info(f"Detected in-batch schema: {columns}")
        else:
            raise ValueError(
                f"Dataset schema not recognized. Expected either ['anchor', 'positive', 'negative'] "
                f"or ['anchor', 'positive']. Found: {columns}"
            )

        # Build the queries
        queries = {f"q{i}": row["anchor"] for i, row in enumerate(eval_dataset)}

        # Build the corpus
        # If we have hard negatives, include them in the corpus pool.
        # If in-batch, the corpus is just all the available positive documents.
        corpus_docs = list(set(eval_dataset["positive"]))
        if is_hard_negative:
            corpus_docs = corpus_docs + list(eval_dataset["negative"])

        corpus = {f"doc{i}": doc for i, doc in enumerate(corpus_docs)}

        # Create a mapping from document text to its new ID for easy lookup
        doc_to_id = {doc: f"doc{i}" for i, doc in enumerate(corpus_docs)}

        # Map each query to its correct positive document
        relevant_docs = {}
        for i, row in enumerate(eval_dataset):
            query_id = f"q{i}"
            positive_doc_text = row["positive"]
            if positive_doc_text in doc_to_id:
                relevant_docs[query_id] = {doc_to_id[positive_doc_text]}

        super().__init__(queries, corpus, relevant_docs, name=name, **kwargs)
        cli_logger.bind(source=type(self).__name__).info(
            f"Evaluation set created with {len(queries)} queries and {len(corpus)} documents in the corpus."
        )


class SeqEvaluator(SequentialEvaluator):
    """An orchestrator subclass that dynamically instantiates sub-evaluators"""

    def __init__(self, evaluators_cfg, main_score_idx=0, eval_dataset=None, **kwargs):
        evaluators = []
        for _eval_key, cfg in evaluators_cfg.items():
            evaluators.append(instantiate(cfg, eval_dataset=eval_dataset, _convert_="none"))

        # Pass the instantiated list to the standard SequentialEvaluator
        super().__init__(
            evaluators=evaluators,
            # Force the Trainer to use the correct metric to save the best model
            main_score_function=lambda scores: scores[main_score_idx],
        )


class STEvaluatorCallback(TrainerCallback):
    def __init__(self, evaluator_fcn: Callable, eval_dataset: Any):
        """Run a SentenceTransformer evaluator as a TrainerCallback. Gets a partial function that yields an evaluator"""
        self.evaluator = evaluator_fcn(eval_dataset)

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Triggered when trainer.evaluate() is called.
        """
        cli_logger.bind(source=type(self).__name__).info(f"Running custom evaluator at step {state.global_step}...")

        # 1. Get the model from kwargs
        #    'kwargs["model"]' is the actual object being optimized by the Trainer.
        train_model = kwargs.get("model")

        # 2. Unwrap DDP / FSDP
        #    In distributed training, 'train_model' is a DDP(SentenceTransformer).
        #    DDP objects do NOT have an .encode() method.
        unwrap_model = train_model
        while hasattr(unwrap_model, "module"):
            unwrap_model = unwrap_model.module

        # 3. Run the Evaluator
        #    We pass the UNWRAPPED model (the actual SentenceTransformer).
        metrics = self.evaluator(unwrap_model, output_path=args.output_dir, epoch=state.epoch, steps=state.global_step)

        cli_logger.bind(source=type(self).__name__).info(f"Evaluator Finished. Result: {metrics}")
