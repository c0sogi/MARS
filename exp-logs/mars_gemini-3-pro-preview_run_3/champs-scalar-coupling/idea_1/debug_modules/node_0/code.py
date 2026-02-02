import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# ==========================================
# 0. Environment Setup & Overrides
# ==========================================

# Suppress warnings
warnings.filterwarnings("ignore")

# Monkeypatch tqdm to suppress progress bars as per requirements
import tqdm


def nop(it, *a, **k):
    return it


tqdm.tqdm = nop

# Import library components
from library.utils import seed_everything, calculate_log_mae, TargetScaler
from library.data_factory import get_dataloaders
from library.model_arch import CouplingPredictor
from library.training_engine import ModelTrainer
from library.config import TRAIN_META_PATH, TYPE_MAP, WORKING_DIR, SUBMISSION_DIR


def run_demonstration():
    print("Starting Library Demonstration...")

    # Set seed for reproducibility
    seed_everything(42)

    # ==========================================
    # 1. Data Loading Demonstration
    # ==========================================
    print("\n--- 1. Data Loading (Debug Mode) ---")

    # Use debug=True and a small sample size for speed
    # batch_size=32 ensures we have multiple batches even with 1000 samples
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True,
        debug_size=1000,
        batch_size=32,
        num_workers=0,  # Use 0 workers to avoid multiprocessing overhead in this quick demo
    )

    # Verify Loaders
    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader should not be empty."

    # Inspect a single batch
    sample_batch = next(iter(train_loader))
    print("Sample Batch Keys:", sample_batch.keys)
    print(f"Batch Size (Graphs): {sample_batch.num_graphs}")

    # Assertions for Batch Structure
    assert hasattr(sample_batch, "x"), "Batch missing node features 'x'"
    assert hasattr(sample_batch, "edge_index"), "Batch missing 'edge_index'"
    assert hasattr(sample_batch, "edge_attr"), "Batch missing 'edge_attr'"
    assert hasattr(sample_batch, "target_pair"), "Batch missing 'target_pair'"
    assert hasattr(sample_batch, "y"), "Batch missing targets 'y'"

    # Check shapes
    # target_pair should be (BatchSize, 2)
    assert sample_batch.target_pair.shape == (sample_batch.num_graphs, 2)
    # y should be (BatchSize, 1)
    assert sample_batch.y.shape == (sample_batch.num_graphs, 1)

    # ==========================================
    # 2. Model Architecture Demonstration
    # ==========================================
    print("\n--- 2. Model Initialization & Forward Pass ---")

    model = CouplingPredictor()

    # Move model to CPU for this quick check (or GPU if available, handled by torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sample_batch = sample_batch.to(device)

    # Perform Forward Pass
    with torch.no_grad():
        output = model(sample_batch)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        sample_batch.num_graphs,
        1,
    ), f"Expected output shape {(sample_batch.num_graphs, 1)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # ==========================================
    # 3. Utility Functions Verification
    # ==========================================
    print("\n--- 3. Utility Verification ---")

    # 3a. Target Scaler
    print("Verifying TargetScaler...")
    # Create a dummy dataframe to test fitting
    dummy_data = {
        "type": ["1JHC", "1JHC", "2JHH", "2JHH"],
        "scalar_coupling_constant": [100.0, 110.0, -10.0, -12.0],
    }
    df_dummy = pd.DataFrame(dummy_data)

    scaler = TargetScaler()
    scaler.fit(df_dummy)

    # Check if stats were computed
    # 1JHC index is 0, 2JHH index is 6 (based on config.py TYPE_MAP)
    idx_1jhc = TYPE_MAP["1JHC"]
    idx_2jhh = TYPE_MAP["2JHH"]

    mean_1jhc = scaler.means[idx_1jhc].item()
    std_1jhc = scaler.stds[idx_1jhc].item()

    expected_mean = 105.0
    expected_std = np.std([100.0, 110.0], ddof=1)  # pandas uses ddof=1 by default

    assert np.isclose(
        mean_1jhc, expected_mean
    ), f"Scaler Mean mismatch: {mean_1jhc} vs {expected_mean}"
    assert np.isclose(
        std_1jhc, expected_std
    ), f"Scaler Std mismatch: {std_1jhc} vs {expected_std}"

    # Test Transform and Inverse Transform
    targets = torch.tensor([105.0], device=device)
    types = torch.tensor([idx_1jhc], device=device)

    scaled = scaler.transform(targets, types)
    # (105 - 105) / std = 0
    assert (
        torch.abs(scaled).item() < 1e-5
    ), "Scaling failed (expected ~0 for mean value)"

    inverted = scaler.inverse_transform(scaled, types)
    assert torch.isclose(
        inverted, targets
    ).item(), "Inverse transform failed to recover original value"

    # 3b. Log MAE Metric
    print("Verifying Log MAE Metric...")
    # Case: Error is e (approx 2.718). Log(e) = 1.
    preds = torch.tensor([2.71828], device=device)
    truth = torch.tensor([0.0], device=device)
    types_t = torch.tensor([0], device=device)

    metric = calculate_log_mae(preds, truth, types_t)
    # log(MAE + 1e-9) ~ log(e) = 1
    print(f"Calculated LogMAE: {metric.item():.4f}")
    assert 0.9 < metric.item() < 1.1, "LogMAE calculation seems incorrect"

    # ==========================================
    # 4. Training Engine Demonstration
    # ==========================================
    print("\n--- 4. Training Engine (Train/Val/Predict) ---")

    # Initialize Trainer
    # Note: TargetScaler inside Trainer will load the actual train.csv.
    # This is fast enough for the demo.
    trainer = ModelTrainer(model, train_loader, val_loader, test_loader)

    # Run 1 Epoch of Training
    print("Running training epoch...")
    train_loss = trainer.train_epoch(epoch_idx=1)
    print(f"Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Run Validation
    print("Running validation...")
    val_metric = trainer.validate()
    print(f"Validation LogMAE: {val_metric:.4f}")
    # Metric can be negative (log of small error), so just check it's a number
    assert isinstance(val_metric, float)

    # Force save the model as 'best_model.pth' so predict works
    # (Usually handled by the run() loop, but we are calling methods individually)
    torch.save(model.state_dict(), trainer.best_model_path)

    # Run Prediction
    print("Running prediction...")
    submission_df = trainer.predict()

    # Verify Submission
    print("Submission DataFrame Head:")
    print(submission_df.head())

    assert "id" in submission_df.columns
    assert "scalar_coupling_constant" in submission_df.columns
    assert len(submission_df) > 0

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not saved."

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
