import logging

from loguru import logger
from transformers import TrainerCallback


class InterceptHandler(logging.Handler):
    """
    Redirects standard logging messages to Loguru.
    Automatically binds the 'source' field to the logger name (e.g., 'transformers').
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Bind 'source'
        logger.bind(source=record.name).opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging_interception(verbose_cli: bool = False):
    """
    Hijacks the standard logging module and redirects everything to Loguru.
    Aggressively removes handlers from libraries (like sentence_transformers)
    that try to manage their own logging.
    """
    # 1. Create our InterceptHandler
    intercept_handler = InterceptHandler()

    # 2. Configure the ROOT logger to send everything to Loguru
    # 'force=True' removes any existing root handlers
    logging.basicConfig(handlers=[intercept_handler], level=0, force=True)

    # 3. Walk through ALL existing loggers (including sentence_transformers, httpx, etc.)
    # and strip their handlers so they use ours.
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)

        # Remove handlers attached directly to this logger (e.g. sentence_transformers)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        # Ensure the logger propagates its messages up to the root (where we catch them)
        logger.propagate = True

    # 4. Specific tweaks for noisy libraries (Optional)
    # httpx and urllib3 can be very chatty, so we raise their level to WARNING
    if not verbose_cli:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("fsspec.local").setLevel(logging.WARNING)


class CLILoggerTrainerCallback(TrainerCallback):
    """
    A HF Trainer Callback that logs metrics to Loguru instead of printing to stdout.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Clean up the logs for readability
            # Remove redundant 'epoch' if present, keeps the line shorter
            logs_copy = logs.copy()
            if "epoch" in logs_copy:
                logs_copy["epoch"] = round(logs_copy["epoch"], 2)

            logger.bind(source="TRAINER-HF").info(f"Step {state.global_step}: {logs_copy}")
