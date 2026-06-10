from typing import Any

from nextgen_train.src.core.components import ComponentFactory
from nextgen_train.src.core.factories import DIFactory
from nextgen_train.src.integrations.sentence_transformers.collators import STCollator
from nextgen_train.src.integrations.sentence_transformers.wrappers import STWrapper


class STModelFactory(ComponentFactory):
    """Builds models as STWrappers"""

    def build(self) -> STWrapper:
        return STWrapper(**self.params)


class STCollatorFactory(ComponentFactory):
    """Builds collators for Sentence Transformers"""

    def build(self) -> STCollator:
        if self.state is None:
            raise ValueError()
        return STCollator(tokenizer=self.state.tokenizer, **self.params)


class STLossFactory(DIFactory):
    """Builds Sentence Transformer Losses."""

    def get_dependencies(self) -> dict[str, Any]:
        """Specify dependencies for sentence transformer losses"""
        if self.state is None:
            raise ValueError()
        st_model = self.state.model.model if self.state.model else None
        st_guide = self.state.guide.model if self.state.guide else None

        return {
            "model": st_model,
            "sentence_embedding_dimension": st_model.get_sentence_embedding_dimension() if st_model else None,
            "guide": st_guide,
        }


class STEvaluatorFactory(DIFactory):
    """Builds Sentence Transformer Evaluators."""

    def get_dependencies(self) -> dict[str, Any]:
        return {"eval_dataset": self.state.eval_dataset}


class STCallbackFactory(DIFactory):
    """Builds callbacks for SentenceTransformerTrainers"""

    def get_dependencies(self) -> dict[str, Any]:
        return {"logger": self.state.logger, "eval_dataset": self.state.eval_dataset}


class STTrainerFactory(DIFactory):
    """Builds trainers for SentenceTransformers"""

    def get_dependencies(self) -> dict[str, Any]:
        return {
            "model": self.state.model.model if self.state.model else None,
            "train_dataset": self.state.train_dataset,
            "eval_dataset": self.state.eval_dataset,
            "data_collator": self.state.collator,
            "callbacks": self.state.callbacks,
            "loss": self.state.loss,
            "evaluator": self.state.evaluator,
        }
