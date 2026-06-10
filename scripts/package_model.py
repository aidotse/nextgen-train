import argparse
import shutil
from pathlib import Path

from loguru import logger
from transformers import AutoConfig


def package_checkpoint(checkpoint_path, hf_model_name):
    ckpt_dir = Path(checkpoint_path).resolve()

    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        logger.error(f"The checkpoint directory '{ckpt_dir}' does not exist.")
        return

    # 1. Setup the /dist and temporary directories
    dist_dir = ckpt_dir / "dist"
    temp_dir = dist_dir / "pruned_model"

    logger.info(f"Creating distribution folder at: {dist_dir}")
    dist_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define exactly what we want to copy
    files_to_keep = [
        "gaudi_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    ]

    # 3. Copy the files safely without modifying the source
    logger.info("Copying inference files...")
    for file_name in files_to_keep:
        src_file = ckpt_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, temp_dir / file_name)
            logger.info(f"Copied: {file_name}")
        else:
            logger.warning(f"'{file_name}' not found in checkpoint. Skipping.")

    # 4. Fetch the missing config.json from Hugging Face
    logger.info(f"Fetching model config for '{hf_model_name}' from Hugging Face Hub...")
    try:
        config = AutoConfig.from_pretrained(hf_model_name, cache_dir=temp_dir)
        config.save_pretrained(temp_dir)
        logger.success("Successfully downloaded and saved config.json")
    except Exception as e:
        logger.error(f"Error fetching config: {e}")
        logger.info("Please ensure the HF model name is correct and you have internet access.")
        exit(1)

    # 5. Zip the temporary directory
    zip_path = dist_dir / f"{Path(checkpoint_path).parent.name}_{Path(checkpoint_path).name}"
    logger.info(f"Zipping contents into {zip_path}.zip...")
    shutil.make_archive(base_name=str(zip_path), format="zip", root_dir=temp_dir)

    # 6. Clean up the temporary staging folder
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    logger.success(f"Your zipped inference model is ready at: {zip_path}.zip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune and package a Gaudi HPU checkpoint for inference.")
    parser.add_argument("checkpoint_path", type=str, help="Path to your Gaudi checkpoint folder")
    parser.add_argument("hf_model_name", type=str, help="Hugging Face model string (e.g., 'meta-llama/Llama-2-7b-hf')")

    args = parser.parse_args()
    package_checkpoint(args.checkpoint_path, args.hf_model_name)
