import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_global_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import Trainer

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


class DemoConfig(Config):
    """
    Custom configuration for the demonstration to ensure speed and isolation.
    """

    def __init__(self):
        super().__init__(debug=True)

        # Use a separate working directory for this demo
        self.working_dir = "./working/demo_execution"

        # Clean up previous demo runs if they exist
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        os.makedirs(self.working_dir, exist_ok=True)

        # Update paths to point to the new working directory
        self.model_save_path = os.path.join(self.working_dir, "best_model.pth")
        self.submission_path = os.path.join(self.working_dir, "submission.csv")

        # Optimization for speed
        self.epochs = 2
        self.debug_subset_size = 50  # Use only 50 samples
        self.batch_size = 8
        self.num_workers = 0  # Avoid multiprocessing overhead for small demo


def verify_data_shapes(loader, name="Loader"):
    """
    Verifies the shapes of data yielded by the dataloader.
    """
    print(f"Verifying {name}...")
    batch = next(iter(loader))
    inputs = batch["inputs"]
    bpp_indices = batch["bpp_indices"]
    ids = batch["id"]

    # Check Inputs: (Batch, SeqLen, InputDim)
    assert inputs.dim() == 3, f"Inputs should be 3D, got {inputs.dim()}"
    assert (
        inputs.shape[1] == 107
    ), f"Sequence length should be 107, got {inputs.shape[1]}"
    assert inputs.shape[2] == 14, f"Input dim should be 14, got {inputs.shape[2]}"

    # Check BPP Indices: (Batch, SeqLen)
    assert bpp_indices.dim() == 2, f"BPP indices should be 2D, got {bpp_indices.dim()}"
    assert bpp_indices.shape[1] == 107, f"BPP sequence length should be 107"

    # Check Targets if present
    if "targets" in batch:
        targets = batch["targets"]
        assert targets.dim() == 3, f"Targets should be 3D, got {targets.dim()}"
        assert targets.shape[1] == 107, f"Target sequence length should be 107"
        assert targets.shape[2] == 5, f"Num targets should be 5"

    print(f"  {name} shapes verified successfully.")


def verify_metric_logic():
    """
    Unit test for the MCRMSE calculation.
    """
    print("Verifying Metric Logic (MCRMSE)...")
    seq_scored = 68
    num_targets = 5

    # Case 1: Perfect prediction
    preds = np.zeros((10, 107, num_targets))
    targets = np.zeros((10, 107, num_targets))
    score = calculate_global_mcrmse(preds, targets, seq_scored=seq_scored)
    assert score == 0.0, f"Score should be 0.0 for perfect prediction, got {score}"

    # Case 2: Constant error of 1.0
    # RMSE of 1.0 is 1.0. Mean of 1.0s is 1.0.
    preds = np.zeros((10, 107, num_targets))
    targets = np.ones((10, 107, num_targets))
    score = calculate_global_mcrmse(preds, targets, seq_scored=seq_scored)

    # Allow small float tolerance
    assert (
        abs(score - 1.0) < 1e-6
    ), f"Score should be 1.0 for constant error, got {score}"

    # Case 3: Error only outside scored region
    # Should result in 0.0 score because we only score first 68 positions
    preds = np.zeros((10, 107, num_targets))
    targets = np.zeros((10, 107, num_targets))
    targets[:, 70:, :] = 100.0  # Huge error outside scored region

    score = calculate_global_mcrmse(preds, targets, seq_scored=seq_scored)
    assert (
        score == 0.0
    ), f"Score should be 0.0 when errors are outside scored region, got {score}"

    print("  Metric logic verified successfully.")


def verify_model_forward_pass(config):
    """
    Verifies that the model can process a batch and produce correct output shapes.
    """
    print("Verifying Model Forward Pass...")
    model = RNAModel(config).to(config.device)
    model.eval()

    # Create dummy batch
    batch_size = 2
    dummy_inputs = torch.randn(batch_size, config.seq_len, config.input_dim).to(
        config.device
    )
    dummy_bpp = torch.randint(0, config.seq_len, (batch_size, config.seq_len)).to(
        config.device
    )

    with torch.no_grad():
        output = model(dummy_inputs, dummy_bpp)

    assert output.shape == (
        batch_size,
        config.seq_len,
        config.num_targets,
    ), f"Output shape mismatch. Expected {(batch_size, config.seq_len, config.num_targets)}, got {output.shape}"

    print("  Model forward pass verified successfully.")


if __name__ == "__main__":
    # 1. Setup Configuration
    print("--- Initializing Configuration ---")
    config = DemoConfig()
    set_seed(config.seed)
    print(f"Working Directory: {config.working_dir}")
    print(f"Device: {config.device}")

    # 2. Data Loading
    print("\n--- Loading Data ---")
    # Note: get_dataloaders handles caching internally within working_dir
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=False
    )

    # Verify Data
    verify_data_shapes(train_loader, "Train Loader")
    verify_data_shapes(val_loader, "Val Loader")
    verify_data_shapes(test_loader, "Test Loader")

    # 3. Model Verification
    print("\n--- Verifying Model Architecture ---")
    verify_model_forward_pass(config)

    # 4. Metric Verification
    print("\n--- Verifying Metric Implementation ---")
    verify_metric_logic()

    # 5. Training Loop
    print("\n--- Starting Training Demo ---")
    trainer = Trainer(config)

    # Run training (fit handles training and validation)
    trainer.fit(train_loader, val_loader)

    # Verify model file creation
    assert os.path.exists(config.model_save_path), "Model file was not saved!"
    print(f"Best model saved at: {config.model_save_path}")

    # 6. Prediction
    print("\n--- Running Prediction ---")
    ids, preds = trainer.predict(test_loader)

    # Verify prediction shape
    # preds shape should be (N_test, 107, 5)
    # In debug mode, N_test is min(240, debug_subset_size) -> 50
    expected_samples = config.debug_subset_size
    assert (
        len(ids) == expected_samples
    ), f"Expected {expected_samples} IDs, got {len(ids)}"
    assert preds.shape == (
        expected_samples,
        107,
        5,
    ), f"Prediction shape mismatch: {preds.shape}"

    # 7. Submission Generation
    print("\n--- Generating Submission ---")
    trainer.generate_submission(ids, preds)

    # Verify submission file
    assert os.path.exists(config.submission_path), "Submission file not found!"

    df_sub = pd.read_csv(config.submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Expected rows: num_samples * seq_len
    expected_rows = expected_samples * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("\n=== Demonstration Completed Successfully ===")
