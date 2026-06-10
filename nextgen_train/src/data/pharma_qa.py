"""Module with abstractions for datasets."""

from datasets import Dataset, load_dataset, load_from_disk
from loguru import logger as cli_logger


class BasePharmaQAData:
    """Base class for PharmaQA datasets."""

    def __init__(
        self, hf_path: str, test_size: str, seed: int, sample_pct: float = 1.0, num_shards: int = 1, shard_idx: int = 0
    ):
        self.hf_path = hf_path
        self.test_size = test_size
        self.seed = seed
        self.sample_pct = sample_pct

        total_dataset = self._load()
        if self.sample_pct < 1.0:
            total_dataset = self._subsample(total_dataset, self.sample_pct, self.seed)
        total_dataset = self._process_dataset(total_dataset)
        split_dataset = total_dataset.train_test_split(test_size=self.test_size, seed=self.seed)
        self._train_dataset = (
            split_dataset["train"].shuffle(seed=seed).shard(num_shards=num_shards, index=shard_idx, contiguous=True)
        )
        self._eval_dataset = split_dataset["test"]
        cli_logger.bind(source=type(self).__name__).info("Created datasets")

    def _load(self) -> Dataset:
        """Load dataset from Hugging Face Hub or local disk."""
        cli_logger.bind(source=type(self).__name__).info(f"Dataset: {self.hf_path}")
        try:
            total_dataset = load_from_disk(self.hf_path)
        except Exception:
            cli_logger.bind(source=type(self).__name__).info("Downloading dataset")
            # I guess we derive train and tests splits from this default split
            total_dataset = load_dataset(self.hf_path, split="train")
        return total_dataset

    @classmethod
    def _subsample(cls, dataset: Dataset, sample_pct: float, seed: int) -> Dataset:
        """Subsample the dataset based on sample_pct."""
        # Calculate number of samples
        new_size = int(len(dataset) * sample_pct)
        cli_logger.bind(source=cls.__name__).info(f"Subsampling dataset to {sample_pct:.1%} ({new_size} samples)...")
        # Shuffle (with seed for reproducibility) and Select
        return dataset.shuffle(seed=seed).select(range(new_size))

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def _process_dataset(self):
        raise NotImplementedError("Subclasses must implement the train_dataset property.")


class MinedNegativesPharmaQAData(BasePharmaQAData):
    @classmethod
    def _process_dataset(cls, total_dataset: Dataset) -> Dataset:
        """Process dataset to ensure correct column names and structure."""
        current_columns = total_dataset.column_names
        # Conditionally rename columns for backward compatibility
        if "text1" in current_columns and "text2" in current_columns:
            cli_logger.bind(source=cls.__name__).info("Detected 'text1'/'text2' columns. Renaming for compatibility.")
            # Ensure the negative column also exists in this format
            if "negative" not in current_columns:
                raise ValueError("Old format ('text1', 'text2') dataset is missing the required 'negative' column.")
            total_dataset = total_dataset.rename_columns({"text1": "anchor", "text2": "positive"})
        # Check if the data is already in the new, correct format
        elif "anchor" in current_columns and "positive" in current_columns:
            if "negative" not in current_columns:
                raise ValueError("New format ('anchor', 'positive') dataset is missing the required 'negative' column.")
            cli_logger.bind(source=cls.__name__).info(
                "Dataset is already in the correct ('anchor', 'positive', 'negative') format."
            )
        # If neither format is found, raise an error
        else:
            raise ValueError(
                f"Dataset columns are incorrect. \
                Expected either ('text1', 'text2', 'negative') or ('anchor', 'positive', 'negative'), \
                but got: {current_columns}"
            )
        columns_to_keep = ["anchor", "positive", "negative"]
        all_columns = total_dataset.column_names
        columns_to_remove = [col for col in all_columns if col not in columns_to_keep]
        if columns_to_remove:
            total_dataset = total_dataset.remove_columns(columns_to_remove)
            cli_logger.bind(source=cls.__name__).info(
                f"Cleaned dataset. Removed extraneous columns: {columns_to_remove}"
            )
        return total_dataset


class InBatchNegativesPharmaQAData(BasePharmaQAData):
    @classmethod
    def _process_dataset(cls, total_dataset: Dataset) -> Dataset:
        """
        Process dataset to ensure correct column names and structure.

        Expect input columns `text1` and `text2`, everything else is ignored.
        These are renamed to `anchor` and `positive` respectively.
        The negatives will be generated in-batch.
        """

        cli_logger.bind(source=cls.__name__).info("Applying safety filter to remove any incomplete pairs...")
        original_size = len(total_dataset)

        total_dataset = total_dataset.filter(
            lambda example: isinstance(example.get("text1"), str) and isinstance(example.get("text2"), str)
        )
        columns_to_keep = ["text1", "text2"]
        columns_to_remove = [col for col in total_dataset.column_names if col not in columns_to_keep]
        total_dataset = total_dataset.remove_columns(columns_to_remove)
        total_dataset = total_dataset.rename_columns({"text1": "anchor", "text2": "positive"})

        new_size = len(total_dataset)
        if original_size != new_size:
            cli_logger.bind(source=cls.__name__).warning(
                f"Safety filter removed {original_size - new_size} incomplete pairs."
            )
        return total_dataset
