from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pytest import approx
from torch.utils.data import Dataset
from transformers import Trainer as HFTrainer
from transformers import TrainingArguments

from nextgen_train.src.core.state import ExperimentState
from nextgen_train.src.integrations.sentence_transformers.wrappers import STWrapper
from nextgen_train.src.nextgen.trainer_fedavg import Trainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Trainable Layer (The one we expect to change)
        self.fc_train = nn.Linear(1, 1, bias=False)

        # 2. Frozen Layer (The one we expect to stay the same)
        self.fc_frozen = nn.Linear(1, 1, bias=False)
        for param in self.fc_frozen.parameters():
            param.requires_grad = False

        # 3. Buffers (BatchNorm stats we need to aggregate)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, input_values, labels=None, num_items_in_batch=None):
        """
        HF Compatible Forward Pass.
        Accepts 'input_values' and 'labels'.
        Returns dict with 'loss' and 'logits'.
        """
        # Pass through layers
        x = self.fc_train(input_values)
        x = self.bn(x)
        # We add the frozen layer just to prove it's part of the graph,
        # even if gradients stop there.
        logits = self.fc_frozen(x)

        loss = None
        if labels is not None:
            # Simple MSE Loss
            loss = (logits - labels).pow(2).mean()

        return {"loss": loss, "logits": logits}


class DummyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        # Returns random data compatible with TinyModel
        return {"input_values": torch.randn(1), "labels": torch.randn(1)}


@pytest.fixture
def trainer_with_bypass2(tmp_path):
    """
    Creates a Trainer instance WITHOUT running __init__.
    We manually inject only what the other unit tests need.
    """
    # Bypass __init__ to avoid loading datasets/configs
    trainer = Trainer.__new__(Trainer)
    trainer.temp_dir = tmp_path

    # Mock trainer node
    mock_node = MagicMock()
    mock_node.client_id = "TEST_ID"
    mock_node.name = f"CLIENT_{mock_node.client_id}"
    trainer.node = mock_node

    # Mock logger. Mock the context manager `with self.logger.start(...)`
    trainer.logger = MagicMock()
    trainer.logger.start.return_value.__enter__.return_value = {"info/run_id": "test_run"}

    # Starting experiment state
    trainer.expstate = ExperimentState(logger=trainer.logger)

    ## Mock everything that is setup by _build_static_assets and _build_computation_graph

    # Model mock
    tiny_model = TinyModel()
    with patch("nextgen_train.src.integrations.sentence_transformers.wrappers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = tiny_model
        wrapper = STWrapper("dummy-model")
        trainer.expstate.model = wrapper
        safe_wrapper_reference = wrapper

    # Mock collator
    collator = MagicMock(spec=["max_length"])
    collator.max_length = 512
    trainer.expstate.collator = collator

    # Mock HF Trainer
    training_args = TrainingArguments(
        output_dir=str(tmp_path / "hf_out"),
        max_steps=1,
        learning_rate=1.0,  # High LR to force weight change
        use_cpu=True,
        report_to="none",
        data_seed=52,
        remove_unused_columns=False,
    )
    trainer.cfg = MagicMock()
    trainer.cfg.model = MagicMock()
    trainer.cfg.collator = MagicMock()
    trainer.cfg.trainer = MagicMock()
    trainer.cfg.trainer.args = OmegaConf.create(training_args.to_dict())
    trainer.cfg.device = "cpu"
    trainer.cfg.stretch_lr = False
    trainer.stretch_lr = trainer.cfg.stretch_lr

    # Define a side_effect function for instantiate
    # Note that different objects here get built at different times, i.e. in the _build_static_assets method or
    # _build_computation_graph method of the original trainer object; only the latter gets called during training!
    def smart_instantiate(config, **kwargs):
        builder = MagicMock()
        if config == trainer.cfg.model:
            tensor_mock = MagicMock()
            tensor_mock.to.return_value = safe_wrapper_reference
            builder.build.return_value = tensor_mock
        elif config == trainer.cfg.collator:
            builder.build.return_value = collator
        elif config == trainer.cfg.trainer:
            hf_trainer = HFTrainer(model=safe_wrapper_reference, args=training_args, train_dataset=DummyDataset())
            trainer.expstate.trainer = hf_trainer
            builder.build.return_value = hf_trainer
        else:
            # Fallback for loss, evaluator, etc.
            builder.build.return_value = MagicMock()
        return builder

    # 5. Apply the Patch
    with patch("nextgen_train.src.nextgen.trainer_fedavg.instantiate", side_effect=smart_instantiate):
        yield trainer


@pytest.fixture
def trainer_with_bypass(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.temp_dir = tmp_path

    ## LAYER 1: STATIC LAYER (persist every round, the global "state" of the Trainer that does not change)
    trainer.node = MagicMock(client_id="TEST_ID", name="CLIENT_TEST_ID")
    trainer.logger = MagicMock()
    trainer.logger.start.return_value.__enter__.return_value = {"info/run_id": "test_run"}
    trainer.expstate = ExperimentState(logger=trainer.logger)
    training_args = TrainingArguments(
        output_dir=str(tmp_path / "hf_out"),
        max_steps=1,
        learning_rate=1.0,
        use_cpu=True,
        report_to="none",
        data_seed=52,
        remove_unused_columns=False,
    )
    trainer.cfg = MagicMock()
    trainer.cfg.trainer.args = OmegaConf.create(training_args.to_dict())
    trainer.cfg.device = "cpu"
    trainer.cfg.stretch_lr = False
    trainer.stretch_lr = trainer.cfg.stretch_lr

    # Test Spies (Objects we hold onto so we can assert against them later)
    with patch("nextgen_train.src.integrations.sentence_transformers.wrappers.SentenceTransformer") as mock_cls:
        mock_cls.return_value = TinyModel()
        safe_wrapper_reference = STWrapper("dummy-model")
        trainer.expstate.model = safe_wrapper_reference

    collator = MagicMock(spec=["max_length"], max_length=512)

    ## LAYER 2: EPHEMERAL LAYER (objects that get torn down and therefore need re-building every train iteration)
    # Since the Trainer re-builds them, we can't set them once and forget about them - we need to override the
    # creation logic
    def smart_instantiate(config, **kwargs):
        builder = MagicMock()

        if config == trainer.cfg.model:
            # Re-inject our spy reference into the new computation graph
            tensor_mock = MagicMock()
            tensor_mock.to.return_value = safe_wrapper_reference
            builder.build.return_value = tensor_mock

        elif config == trainer.cfg.collator:
            builder.build.return_value = collator

        elif config == trainer.cfg.trainer:
            fresh_hf_trainer = HFTrainer(model=safe_wrapper_reference, args=training_args, train_dataset=DummyDataset())
            builder.build.return_value = fresh_hf_trainer

        else:
            builder.build.return_value = MagicMock()

        return builder

    # Apply the factory patch
    with patch("nextgen_train.src.nextgen.trainer_fedavg.instantiate", side_effect=smart_instantiate):
        yield trainer


def test_setup(trainer_with_bypass):
    trainer = trainer_with_bypass
    result_path = trainer.setup()

    # 1. File Existence
    assert result_path.exists()
    assert result_path.name == "global_model_v0.pth"

    # 2. Key Correctness (The "Prefix" Check)
    # We load the file to prove it contains the UnifiedTinyModel weights
    saved_state = torch.load(result_path)

    # Check for a known key from UnifiedTinyModel
    assert "fc_train.weight" in saved_state, "The file doesn't contain the model weights!"

    # Check that we didn't accidentally save wrapper keys (e.g. _inner_model...)
    # This confirms setup() called the correct save method on the wrapper.
    for key in saved_state.keys():
        assert not key.startswith(
            "_inner_model"
        ), f"Bug detected: Saved key '{key}' has wrapper prefix. setup() likely bypassed the wrapper."


def test_train(trainer_with_bypass):
    trainer = trainer_with_bypass
    tmp_path = trainer.temp_dir

    # --- STEP 1: Create Initial State on Disk ---
    # Simulate the server sending us a model with specific weights (e.g., 10.0)
    initial_weights = trainer.expstate.model.model.state_dict()
    nn.init.constant_(initial_weights["fc_train.weight"], 10.0)

    server_model_path = tmp_path / "global_model.pth"
    torch.save(initial_weights, server_model_path)

    # --- STEP 2: Execute 'train' ---
    # This triggers: Load(10.0) -> Train(Update) -> Save(New)
    output_path = trainer.train(server_model_path, "0.0.0")

    # --- STEP 3: Verification ---

    # check File Output
    assert output_path.exists()
    assert output_path.name == "local_model.pth"

    # check Weight Update
    final_state = torch.load(output_path)
    final_weight = final_state["fc_train.weight"].item()

    # assert weights changed (Training happened)
    assert final_weight != 10.0, "Weights remained static. Did HF Trainer run?"

    # assert weights didn't explode (Basic sanity check)
    assert abs(final_weight) < 100.0


def test_reduce_models(trainer_with_bypass):
    trainer = trainer_with_bypass
    temp_dir = trainer.temp_dir

    # Helper to generate a client state based on the baseline
    def make_client_state(train_weight, run_mean, batches, frozen_weight=None):
        state = trainer.expstate.model.model.state_dict()
        state["fc_train.weight"] = torch.tensor([[train_weight]])
        state["bn.running_mean"] = torch.tensor([run_mean])
        state["bn.num_batches_tracked"] = torch.tensor(batches)
        if frozen_weight is not None:
            state["fc_frozen.weight"] = torch.tensor([[frozen_weight]])
        return state

    # --- SETUP: Server Baseline ---
    nn.init.constant_(trainer.expstate.model.model.fc_train.weight, 1.0)
    nn.init.constant_(trainer.expstate.model.model.fc_frozen.weight, 1.0)
    nn.init.constant_(trainer.expstate.model.model.bn.running_mean, 0.0)
    trainer.expstate.model.model.bn.num_batches_tracked.fill_(0)

    # --- SETUP: Clients ---
    path_a = temp_dir / "client_a.pth"
    torch.save(make_client_state(3.0, 10.0, 5, frozen_weight=999.0), path_a)

    path_b = temp_dir / "client_b.pth"
    torch.save(make_client_state(5.0, 20.0, 10), path_b)

    # --- ACTION: Call the real method ---
    # To run this, ensure the body of reduce_models is pasted into the Trainer class above
    # or imported from your actual module.
    result_path = trainer.reduce_models([path_a, path_b], "0.0.0")

    # --- ASSERTIONS ---
    result_state = torch.load(result_path)

    # 1. Weights averaged (3+5)/2 = 4.0
    assert result_state["fc_train.weight"].item() == approx(4.0)

    # 2. Frozen weights preserved (Ignored Client A's 999.0, kept Server's 1.0)
    assert result_state["fc_frozen.weight"].item() == approx(1.0)

    # 3. Buffers averaged (10+20)/2 = 15.0
    assert result_state["bn.running_mean"].item() == approx(15.0)

    # 4. Integer Buffers used MAX logic (max(5, 10) = 10)
    assert result_state["bn.num_batches_tracked"].item() == approx(10)
