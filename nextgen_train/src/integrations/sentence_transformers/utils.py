import types
from contextlib import contextmanager

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


@contextmanager
def safe_encode_context(model: SentenceTransformer, max_len: int, batch_size: int = 32):
    """
    Safely executes sentence-transformer encode operations across any hardware backend
    (CPU, NVIDIA CUDA, AMD ROCm, Intel Gaudi HPU).

    This patches `model.encode` to enforce two strict behaviors:
    1. Enforces static tensor shapes (padding='max_length') to prevent hardware graph
       recompilation crashes on XLA/HPU devices.
    2. Forces a final cast to float32, preventing numpy conversion crashes during
       Distributed Data Parallel (DDP) runs operating in bfloat16 mixed-precision.

    Designed specifically for SentenceTransformer models.
    """
    original_encode = model.encode

    def safe_encode(
        model_self, sentences: str | list[str], batch_size_arg: int = 32, **kwargs
    ) -> torch.Tensor | np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        # Use the provided fixed batch size to ensure static graphs if on HPU
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]

            # STATIC TOKENIZATION (Crucial)
            # We force padding='max_length' so every tensor is [1, self.max_len].
            # This prevents the HPU Graph from crashing on variable lengths.
            features = model_self.tokenizer(
                batch_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
            )
            features = {k: v.to(model_self.device) for k, v in features.items()}

            # MANUAL FORWARD: Bypasses 'wrapped_hpugraph_forward' wrappers
            # NOTE: this runs when the evaluator calls the method later, do not remove the context
            with torch.no_grad():
                for module in model_self:
                    features = module(features)

            # Collect result
            # SentenceTransformers stores the embedding in 'sentence_embedding'
            emb = features["sentence_embedding"]
            all_embeddings.append(emb.cpu())

        # Format Output
        # Concatenate and return as numpy (standard for Evaluator)
        # In DDP apparently casting isnt automatic - Force cast to float32 to survive Gaudi BF16/Mixed Precision
        all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
        return all_embeddings_tensor.to(torch.float32).numpy()

    # Apply the patch
    model.encode = types.MethodType(safe_encode, model)
    try:
        yield model
    finally:
        # Safely restore the original encode method
        model.encode = original_encode
