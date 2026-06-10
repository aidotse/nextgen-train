"""An experiment logger using MLFlow experiment tracking API.

To start the server (example):

poetry run mlflow server --backend-store-uri /mnt/pr_honeypot/mlruns/ --port 5000
"""

import abc
import copy
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric
from omegaconf import DictConfig, open_dict
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from nextgen_train.src.utils.util import flatten_dict, get_git_rev_hash, get_working_tree_hash


class ExperimentLogger(abc.ABC):
    """A generic experiment logger"""

    @abc.abstractmethod
    @contextmanager
    def start(self, config: DictConfig) -> Generator[dict, None, None]:
        pass

    @abc.abstractmethod
    def log_artifact(self, artifact_path: str) -> None:
        """Logs an artifact, registering it to the current run."""

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to the current run."""

    @abc.abstractmethod
    def set_tag(self, key: str, value: Any) -> None:
        """Sets a tag for the current run."""

    @abc.abstractmethod
    def set_global_step_offset(self, offset: int) -> None:
        """Sets the global step offset for resuming runs."""

    @abc.abstractmethod
    def get_global_step_offset(self) -> int:
        """Gets the global step offset for resuming runs."""

    @staticmethod
    @abc.abstractmethod
    def clean_key(key: str) -> str:
        """Key sanitization for logging"""

    @staticmethod
    @abc.abstractmethod
    def clean_val(val: Any) -> Any:
        """Value sanitization for logging"""


class MLFlowTrainerCallback(TrainerCallback):
    """A wrapper for an existing logger to use with Hugging Face's Trainer as a callback."""

    def __init__(self, logger: ExperimentLogger | None = None):
        self.logger = logger

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict[str, float], **kwargs
    ) -> None:
        """
        Event called after logging the last logs.
        `logs` contains the metrics that were just logged by the Trainer's internal mechanism.
        For training steps e.g.: {'loss': ..., 'learning_rate': ..., 'epoch': ...}
        For evaluation steps e.g.: {'eval_loss': ..., 'eval_bleu': ..., 'eval_runtime': ..., 'epoch': ...}
        """
        if self.logger and logs:
            # Sanitize log strings
            clean_logs = {}
            for key, val in logs.items():
                value = self.logger.clean_val(val)
                new_key = self.logger.clean_key(key)
                clean_logs[new_key] = value

            self.logger.log_metrics(metrics=clean_logs, step=state.global_step)


class MLFlowLogger(ExperimentLogger):
    """
    Logs metrics and artifacts to an MLFlow backend. Also supports periodic syncing of log files as artifacts
    for real-time monitoring.

    This object sets the tracking URI to an `mlruns` endpoint. We prefer this over logging directly to
    a registry (like Gitlab) because we get nicer visualization features. Better separation of concerns.
    """

    def __init__(
        self,
        mlruns: str,
        flatten_sep: str = "/",
        experiment: str | None = None,
        run: str | None = None,
    ) -> None:
        """Setup the MLFlow logger with some info about the run, experiment, etc.

        Args:
            run_id: If None, create a new run. Otherwise, use this run id, assuming it already exists.
            Throw error if it doesn't.
        """
        self.mlruns = mlruns
        self.client = MlflowClient(tracking_uri=mlruns)
        self.experiment = experiment if experiment else "default"
        self.run_name = run
        self.run_id = None
        self.flatten_sep = flatten_sep
        self.state_metric_key = "system/global_step_marker"
        self._sync_threads: list[BackgroundLogSyncer] = []

    @contextmanager
    def start(self, config: DictConfig, log_sync_paths: list[Path] | Path | None = None) -> Generator[dict, None, None]:
        """Sets up tracking as a context manager.

        Logs the configuration and returns the active run, generating a new experiment and run if required.
        """

        # NOTE: this will OVERWRITE any previously set tracking URI for the duration of the context!
        mlflow.set_tracking_uri(self.mlruns)
        if self.experiment:
            mlflow.set_experiment(self.experiment)
        # This call is what actually creates the run and experiment in MLflow
        with mlflow.start_run(run_id=self.run_id, run_name=self.run_name) as active_run:
            # Also log this run id and experiment name
            self.run_id = active_run.info.run_id if self.run_id is None else self.run_id
            # Only log parameters if config is not empty
            if config:
                with open_dict(config):
                    if "logger" not in config:
                        config.logger = {}
                    config.logger.run = self.run_id
                    config.logger.run_name = active_run.info.run_name
                    config.logger.experiment_id = active_run.info.experiment_id
                # Log call params
                params = dict(copy.deepcopy(config._content))
                mlflow.log_params(flatten_dict(params, sep=self.flatten_sep))
                rev_hash = get_git_rev_hash()
                tree_hash = get_working_tree_hash()
                mlflow.log_param("head", rev_hash if rev_hash is not None else "unknown")
                mlflow.log_param("tree", tree_hash if tree_hash is not None else "unknown")

            # Enable log sync to backend if requested
            if log_sync_paths:
                self.start_sync(log_sync_paths)

            try:
                yield flatten_dict(active_run.to_dictionary())
            finally:
                # Need to stop sync inside the run still
                if log_sync_paths:
                    self.stop_sync()

    @staticmethod
    def clean_key(key: str) -> str:
        """Key sanitation for logging to backend."""
        new_key = key.replace("(", "").replace(")", "")
        new_key = new_key.replace("@", "_at_")
        new_key = new_key.strip()
        return new_key

    @staticmethod
    def clean_val(val: Any) -> Any:
        """Key sanitation for logging to backend."""
        value = val.item() if hasattr(val, "item") else val
        return value

    def start_sync(self, log_file_paths: list[Path] | Path, interval_seconds: int = 5):
        """
        Starts a background thread that periodically uploads logs.
        """
        # Safety: Stop any existing thread before starting a new one
        self.stop_sync()

        # Normalize to list
        if not isinstance(log_file_paths, list):
            log_file_paths = [log_file_paths]

        if self.run_id:
            for path in log_file_paths:
                thread = BackgroundLogSyncer(
                    client=self.client, run_id=self.run_id, log_file=path, interval_seconds=interval_seconds
                )
                self._sync_threads.append(thread)
                thread.start()

    def stop_sync(self):
        """
        Stops the background thread and triggers a final upload.
        """
        for thread in self._sync_threads:
            thread.stop()
        self._sync_threads.clear()

    def log_artifact(self, artifact_path: str) -> None:
        self.client.log_artifact(self.run_id, artifact_path)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        final_step = step + self.get_global_step_offset()
        timestamp = int(time.time() * 1000)  # MLflow requires milliseconds
        metric_objs = [
            Metric(key=self.clean_key(k), value=self.clean_val(v), timestamp=timestamp, step=final_step)
            for k, v in metrics.items()
        ]
        self.client.log_batch(self.run_id, metrics=metric_objs)

    def set_tag(self, key: str, value: Any) -> None:
        self.client.set_tag(self.run_id, key, value)

    def set_global_step_offset(self, offset: int) -> None:
        if not self.run_id:
            raise ValueError("Cannot set global step offset before starting a run.")
        self.client.log_metric(self.run_id, key=self.state_metric_key, value=float(offset), step=offset)

    def get_global_step_offset(self) -> int:
        """Fetches the last recorded step from the MLflow backend."""
        if not self.run_id:
            raise ValueError("Cannot set global step offset before starting a run.")

        run = self.client.get_run(self.run_id)
        # The .data.metrics dict contains only the LATEST value for each key.
        # We use .get(key, 0) to default to 0 if this is a fresh run.
        last_step = run.data.metrics.get(self.state_metric_key, 0)
        return int(last_step)


class BackgroundLogSyncer(threading.Thread):
    def __init__(self, client: MlflowClient, run_id: str, log_file: Path, interval_seconds: int = 5):
        super().__init__(daemon=True)
        self.client = client
        self.run_id = run_id
        self.log_file = log_file
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        # State tracking for optimization
        self.last_mtime = 0.0
        self.last_size = 0

    def run(self):
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.interval):
                break
            self.check_and_upload()

        # Final upload on exit (force it)
        self.upload()

    def check_and_upload(self):
        """
        Checks file metadata. Only uploads if the file has been modified.
        """
        try:
            if not self.log_file.exists():
                return

            # fast OS call (stat) - extremely cheap
            stat = self.log_file.stat()
            current_mtime = stat.st_mtime
            current_size = stat.st_size

            # Only upload if the file is newer OR larger than before
            # (Checking size handles edge cases where mtime resolution is low)
            if current_mtime > self.last_mtime or current_size != self.last_size:
                self.upload()
                self.last_mtime = current_mtime
                self.last_size = current_size

        except Exception:  # noqa: S110
            pass

    def upload(self):
        try:
            # We use the client directly to bypass global state issues
            self.client.log_artifact(self.run_id, str(self.log_file))
        except Exception:  # noqa: S110
            pass

    def stop(self):
        self.stop_event.set()
        self.join()
