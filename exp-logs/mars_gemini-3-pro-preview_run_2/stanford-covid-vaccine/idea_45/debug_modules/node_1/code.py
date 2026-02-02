import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import provided library modules
from library.loss_metric import MCRMSELoss, GlobalMetricsTracker
from library.model_architecture import RDFRN
from library.data_processor import get_dataloaders, get_data
from library.trainer import run_training


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def test_loss_metric():
    print("\n=== Testing Loss Metric (MCRMSELoss) ===")
    criterion = MCRMSELoss()

    # Create dummy predictions and targets
    # Shape: (Batch, Seq_Len, 5)
    batch_size = 4
    seq_len = 107

    preds = torch.rand((batch_size, seq_len, 5), dtype=torch.float32)
    targets = torch.rand((batch_size, seq_len, 5), dtype=torch.float32)

    loss = criterion(preds, targets)

    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert isinstance(loss, torch.Tensor), "Loss must be a torch Tensor"
    assert loss.dim() == 0, "Loss must be a scalar"
    assert loss.item() >= 0, "Loss must be non-negative"

    # Test GlobalMetricsTracker
    print("Testing GlobalMetricsTracker...")
    tracker = GlobalMetricsTracker()
    tracker.update(preds, targets)
    metrics = tracker.compute()

    assert "mcrmse" in metrics, "Tracker must compute mcrmse"
    assert "rmse_reactivity" in metrics, "Tracker must compute rmse_reactivity"
    print("GlobalMetricsTracker check passed.")


def test_model_architecture():
    print("\n=== Testing Model Architecture (RDFRN) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RDFRN().to(device)
    model.eval()

    # Create dummy inputs
    # Input dim is 18 (Seq(4) + Struct(3) + Loop(7) + PartnerID(4))
    batch_size = 2
    seq_len = 107
    input_dim = 18

    inputs = torch.randn((batch_size, seq_len, input_dim)).to(device)
    # Partner indices: random integers between -1 and seq_len-1
    partner_indices = torch.randint(-1, seq_len, (batch_size, seq_len)).to(device)

    with torch.no_grad():
        y1, y2 = model(inputs, partner_indices)

    print(f"Model Output Shapes: y1={y1.shape}, y2={y2.shape}")

    # Assertions
    expected_shape = (batch_size, seq_len, 5)
    assert (
        y1.shape == expected_shape
    ), f"y1 shape mismatch. Expected {expected_shape}, got {y1.shape}"
    assert (
        y2.shape == expected_shape
    ), f"y2 shape mismatch. Expected {expected_shape}, got {y2.shape}"
    print("Model architecture check passed.")


def test_data_pipeline():
    print("\n=== Testing Data Pipeline ===")
    # Use a small batch size for quick verification
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=4, num_workers=0, load_cached_data=False
    )

    # Fetch one batch from train_loader
    batch = next(iter(train_loader))

    inputs = batch["inputs"]
    partner_indices = batch["partner_indices"]
    targets = batch["targets"]
    ids = batch["id"]

    print(f"Batch Keys: {batch.keys()}")
    print(f"Inputs Shape: {inputs.shape}")
    print(f"Targets Shape: {targets.shape}")

    # Assertions
    assert inputs.dim() == 3, "Inputs should be 3D (B, L, C)"
    assert inputs.shape[1] == 107, "Sequence length should be 107"
    assert inputs.shape[2] == 18, "Feature dimension should be 18"
    assert targets.shape[2] == 5, "Target dimension should be 5"
    assert len(ids) == inputs.shape[0], "Number of IDs should match batch size"

    print("Data pipeline check passed.")


def test_full_training_flow():
    print("\n=== Testing Full Training Flow ===")

    # Define output paths
    working_dir = "./working"
    submission_dir = "./submission"
    submission_file = os.path.join(submission_dir, "submission.csv")

    # Clean up previous runs if any (optional, but good for robust testing)
    if os.path.exists(submission_file):
        os.remove(submission_file)

    # Run training for 1 epoch with small batch size to ensure speed
    # load_cached_data=True allows using pre-computed npz files if they exist, speeding up loading
    run_training(epochs=1, batch_size=16, load_cached_data=True)

    # Verify outputs
    assert os.path.exists("./working/best_model.pth"), "Best model file was not saved."
    assert os.path.exists(submission_file), "Submission file was not generated."

    # Check submission content
    df_sub = pd.read_csv(submission_file)
    print(f"Submission File Shape: {df_sub.shape}")
    print(f"Submission Columns: {df_sub.columns.tolist()}")

    # Expected rows: 240 test samples * 107 positions = 25680
    # However, the provided test.json has 240 lines.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, found {len(df_sub)}"

    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match requirements."

    print("Full training flow check passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seeds(42)

    # 1. Test Metric
    test_loss_metric()

    # 2. Test Model
    test_model_architecture()

    # 3. Test Data Loading
    test_data_pipeline()

    # 4. Test Training & Inference
    test_full_training_flow()

    print("\nAll demonstrations completed successfully.")
