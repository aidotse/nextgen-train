import logging
import os
import sys
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

DATA_DIR = Path("./data")
INPUT_FILE = DATA_DIR / "generated_questions_to_data.jsonl"
INPUT_FILE_AZ = DATA_DIR / "generated_questions_from_parquet.jsonl"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Configuration ---
HOME_DIR = Path.home()
SUBSET_SAVE_DIR = Path(DATA_DIR)
SEED = 42


def truncate_by_words(example, max_words=512):
    """A robust function that handles non-string data by converting it to a string first."""
    text1 = example.get("text1")
    text2 = example.get("text2")
    example["text1"] = " ".join(str(text1).split()[:max_words])
    example["text2"] = " ".join(str(text2).split()[:max_words])
    return example


DATA_CONFIG = {
    "questions_answers": {
        "path": str(INPUT_FILE),
        "file_type": "json",
        "column_map": {"question": "text1", "answer": "text2"},
    },
    "questions_answers_fineweb": {
        "path": str(INPUT_FILE_AZ),
        "file_type": "json",
        "column_map": {"question": "text1", "answer": "text2"},
    },
}


def load_and_prepare_dataset(name: str, config: dict) -> Dataset:
    """
    Loads a single dataset from config, renames columns, and returns it.
    """
    logging.info(f"--- Loading and preparing dataset: {name} ---")

    try:
        path = config["path"]
        file_type = config.get("file_type")

        # NOTE: Concatenation requires datasets to be loaded into memory, so streaming is not used here.
        ds = load_dataset(file_type, data_files=path, split="train")

        # Rename columns to a standard format ("text1", "text2")
        if config["column_map"]:
            ds = ds.rename_columns(config["column_map"])

        logging.info(f"Successfully loaded and prepared '{name}' with {len(ds)} rows.")
        return ds

    except Exception as e:
        logging.error(f"Failed to load or prepare {name}. Error: {e}", exc_info=True)
        return None


# --- Main Script Execution ---
if __name__ == "__main__":
    SUBSET_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and prepare all datasets individually
    list_of_datasets = []
    for name, conf in DATA_CONFIG.items():
        dataset = load_and_prepare_dataset(name, conf)
        if dataset:
            list_of_datasets.append(dataset)

    if not list_of_datasets:
        logging.error("No datasets were loaded. Exiting.")
        sys.exit(1)

    # 2. Combine the prepared datasets into one
    logging.info(f"Combining {len(list_of_datasets)} datasets...")
    # Use the `concatenate_datasets` function
    combined_dataset = concatenate_datasets(list_of_datasets)
    logging.info(f"Combined dataset now has {len(combined_dataset)} rows.")

    # 3. Process the single combined dataset
    logging.info("Shuffling and cleaning the combined dataset...")

    shuffled_dataset = combined_dataset.shuffle(seed=SEED)

    # Filter out rows with empty fields and select final columns
    clean_dataset = shuffled_dataset.filter(
        lambda ex: ex.get("text1") and ex.get("text2"),
        num_proc=os.cpu_count() or 4,  # Use available cores, with a fallback
    )
    final_dataset = clean_dataset.select_columns(["text1", "text2"])
    final_dataset = final_dataset.map(truncate_by_words, num_proc=os.cpu_count() or 4)

    logging.info(f"After filtering, {len(final_dataset)} rows remain.")

    # 4. Save the final combined dataset to disk
    output_path = SUBSET_SAVE_DIR / "combined_questions_answers"
    logging.info(f"Saving final combined dataset to: {output_path}")
    final_dataset.save_to_disk(str(output_path))

    logging.info("--- Script finished successfully! ---")
