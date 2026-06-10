import torch.nn as nn
from sentence_transformers import SentenceTransformer

from nextgen_train.src.core.wrappers import BaseWrapper


class STWrapper(BaseWrapper):
    """Adapter for sentence-transformers, adding in additional useful hooks."""

    def __init__(self, model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
        super().__init__()
        self._model = SentenceTransformer(model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def model_max_length(self) -> None | int:
        return self._model.max_seq_length

    @property
    def tokenizer(self) -> None | nn.Module:
        return self._model.tokenizer
