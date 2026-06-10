from typing import Any

import torch


class STCollator:
    """
    A collator suitable for SentenceTransfomerTrainer:

    https://github.com/huggingface/sentence-transformers/blob/d0df0cccfc5564ed24092ae3fe24290060a0f96e/sentence_transformers/trainer.py#L499
    """

    def __init__(self, tokenizer: Any, columns: list[str], valid_label_columns: list[str], max_length=512):
        """
        Args:
            tokenizer: The HF Tokenizer (lightweight).
            max_length: The cut-off length (critical for preventing OOM).
            columns: List of keys to tokenize in order (e.g. ['query', 'doc']).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.columns = columns
        self.valid_label_columns = valid_label_columns

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        """
        Takes an input features batch from a dataloader and transforms it into the standard input format for
        SentenceTransformerTrainer. This is a "flat" sequence of specially-labeled samples.
        """

        # 1. Prepare the Flat Dictionary
        flat_batch = {}

        # 2. Iterate and Flatten
        for i, col in enumerate(self.columns):
            texts = [example[col] for example in batch]

            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 3. Map to "sentence_{i}_{key}" protocol
            #    e.g. sentence_0_input_ids, sentence_1_attention_mask
            for key, tensor in tokenized.items():
                flat_batch[f"sentence_{i}_{key}"] = tensor

        # 4. Handle Labels
        if "label" in batch[0]:
            flat_batch["label"] = torch.tensor([x["label"] for x in batch], dtype=torch.float)

        return flat_batch
