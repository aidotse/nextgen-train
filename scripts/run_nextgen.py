import hashlib
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import hydra
from coolname import generate_slug
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger as cli_logger
from omegaconf import OmegaConf, open_dict

from nextgen_train.src.core.state import ExperimentState
from nextgen_train.src.logging.utils import setup_logging_interception

# Load environment variables from .env file if present
load_dotenv(override=True)

# Enable resolving variables in the configs
OmegaConf.register_new_resolver("eval", eval)

# For DDP gating
global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# CLI formatting
LOG_FORMAT = "<green>{time}</green> | <level>{level: <8}</level> | <cyan>{extra[source]}</cyan> | {message}"
cli_logger.remove()

if global_rank == 0:
    cli_logger.add(
        lambda message: print(message, end=""),
        format=LOG_FORMAT,
        filter=lambda record: "source" in record["extra"],
        colorize=True,
    )
else:
    # Add a strict handler that ONLY allows messages with 'dist_debug=True' for all ranks other than 0
    cli_logger.add(
        lambda message: print(message, end=""),
        format=LOG_FORMAT,
        filter=lambda record: record["extra"].get("dist_debug") is True and "source" in record["extra"],
        colorize=True,
    )


@hydra.main(config_path="../nextgen_train/configs", config_name="nextgen", version_base=None)
def main(cfg):
    """Entrypoint for NextGen integration. Sets up logging and runs the experiment."""

    # Activate file logging intercepts - must happen inside the hydra.main to ensure that it does not get overridden
    setup_logging_interception(cfg.verbose_cli)

    cli_logger.bind(source="MAIN").info("--- Starting Framework Run ---")

    # We expect these variables to be set for access to the Gitlab MLFlow API.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    tracking_token = os.environ.get("MLFLOW_TRACKING_TOKEN", None)
    if not tracking_uri or not tracking_token:
        raise ValueError("Missing either MLFLOW_TRACKING_URI or MLFLOW_TRACKING_TOKEN")

    # Default path for logs
    Path("logs").mkdir(exist_ok=True)

    # Get shared ID
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    shared_id = os.environ.get("DDP_SHARED_ID", None)
    if shared_id is None and world_size > 1:
        raise OSError("Missing DDP_SHARED_ID env variable, used to coordinate logging when using DDP.")

    # Case 1: user provides run name, then take it
    if cfg.run_name:
        run_name = cfg.run_name
    # Case 2: DDP. Deterministically generate the same run name on all ranks
    elif world_size > 1:
        # Hash the shared_id into an integer seed
        seed_int = int(hashlib.sha256(shared_id.encode("utf-8")).hexdigest(), 16) % (2**32)
        # Save the current random state, seed it, generate name, and restore state (!)
        rng_state = random.getstate()
        random.seed(seed_int)
        run_name = generate_slug(2)
        random.setstate(rng_state)
    # Case 3: no DDP and no predefined name -> generate
    else:
        run_name = generate_slug(2)

    # Match run naming back to config
    with open_dict(cfg):
        cfg.run_name = run_name
        cfg.cli_call = " ".join(sys.argv)

    # Log naming
    base_name = f"{Path(__file__).stem}_{run_name}"
    log_file_name = f"logs/{base_name}_rank_{global_rank}.log"

    # CLI logging
    cli_logger.add(
        log_file_name,
        backtrace=True,
        diagnose=True,
        format=LOG_FORMAT,
        filter=lambda record: "source" in record["extra"],
    )

    ## Backend logger
    logger = None
    all_sync_paths = None
    if global_rank == 0:
        # Match log naming to backend logger
        if cfg.get("logger"):
            cfg.logger.run = run_name
            cli_logger.bind(source="MAIN").info("Set logger.run to match NextGen run.")

        logger = instantiate(cfg.get("logger"), state=ExperimentState(), _convert_="none")
        logger = logger.build() if logger else None

        # Rank 0 knows exactly what the files will be named for ALL ranks
        all_sync_paths = [Path(f"logs/{base_name}_rank_{i}.log") for i in range(world_size)]

    # Runner
    runner = instantiate(cfg.framework.framework_runner)

    # Run experiment
    run_context = logger.start(cfg, log_sync_paths=all_sync_paths) if logger else nullcontext()
    with run_context:
        runner.run(cfg, logger, tracking_uri=tracking_uri)

        if global_rank == 0:
            cli_logger.bind(source="MAIN").info("--- Framework Run Finished ---")


if __name__ == "__main__":
    main()
