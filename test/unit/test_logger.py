import os
from pathlib import Path

import mlflow
import pytest
from omegaconf import OmegaConf

from nextgen_train.src.logging.mlflow import MLFlowLogger


@pytest.fixture
def artifact_path(tmp_path):
    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("Example artifact")
    yield Path(tmp_path) / "artifact.txt"


@pytest.fixture(params=["test_experiment_1", "test_experiment_2", "test_experiment_3"])
def logger(tmp_path, request):
    return MLFlowLogger(mlruns=str(Path(tmp_path) / "mlruns"), experiment=request.param, run=None, flatten_sep="/")


def test_logger_context_manager(logger: MLFlowLogger, artifact_path):
    """Test the logger as a context manager."""

    # Create new run
    with logger.start(OmegaConf.create({"param_to_log": "test_param"})) as run_info:
        # Check that the active run belongs to the experiment we specified
        active_run = mlflow.active_run()
        experiment = mlflow.get_experiment_by_name(logger.experiment)
        run_id = run_info["info/run_uuid"]
        query = f"run_id = '{active_run.info.run_id}'"
        results = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=query)
        # Check that the active run is the one we started
        assert active_run.info.run_id == run_id
        assert active_run.info.experiment_id == experiment.experiment_id
        # Check results contents
        assert len(results) == 1
        assert results["run_id"][0] == active_run.info.run_id
        assert results["experiment_id"][0] == experiment.experiment_id
        assert results["status"][0] == "RUNNING"
        sep = logger.flatten_sep
        assert results[f"params.logger{sep}run"][0] == active_run.info.run_id == run_id
        assert results[f"params.logger{sep}run_name"][0] == active_run.info.run_name
        assert results[f"params.logger{sep}experiment_id"][0] == experiment.experiment_id
        assert results["params.param_to_log"][0] == "test_param"
        # Test logging artifact
        logger.log_artifact(str(artifact_path))
        path = results["artifact_uri"][0]
        path = path[7:] if path.startswith("file://") else path
        assert os.path.isfile(Path(path) / "artifact.txt")

    # Resuming existing (last) run
    logger.run_name = active_run.info.run_name
    with logger.start(OmegaConf.create({"another_param": "test_param_2"})) as run_id:
        assert mlflow.active_run().info.run_id == active_run.info.run_id
        query = f"run_id = '{active_run.info.run_id}'"
        results = mlflow.search_runs(experiment_ids=[experiment.experiment_id], filter_string=query)
        assert len(results) == 1
        assert results["params.another_param"][0] == "test_param_2"
