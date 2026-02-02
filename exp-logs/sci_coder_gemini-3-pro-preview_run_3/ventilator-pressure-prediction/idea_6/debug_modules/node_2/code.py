import os
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library
from library.config import Config, seed_everything
from library.features import prepare_datasets
from library.dataset import get_dataloaders
from library.model import PCANet, MaskedMAELoss
from library.train import train_model

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_data(source_path, dest_path, num_breaths=100, seq_len=80):
    """
    Reads the first N breaths from the source CSV and saves to dest_path.
    Ensures that we read exact multiples of seq_len to maintain breath integrity.
    """
    rows_to_read = num_breaths * seq_len
    # Read the header first to ensure we get columns right, then read data
    df = pd.read_csv(source_path, nrows=rows_to_read)

    # Verify we didn't cut a breath in half (check breath_id of last row)
    # In this dataset, breaths are grouped.
    unique_breaths = df["breath_id"].unique()
    if len(unique_breaths) < num_breaths:
        print(
            f"Warning: Requested {num_breaths} breaths but file only had {len(unique_breaths)}."
        )

    df.to_csv(dest_path, index=False)
    print(
        f"Created subset: {dest_path} with {len(df)} rows ({df['breath_id'].nunique()} breaths)."
    )


def main():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup
    seed_everything(42)

    # Define temporary directories for the demo
    demo_input_dir = "./working/demo_input"
    demo_working_dir = "./working/demo_working"
    demo_submission_dir = "./working/demo_submission"

    os.makedirs(demo_input_dir, exist_ok=True)
    os.makedirs(demo_working_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # 2. Create Data Subsets (Optimize for Speed)
    print("\n[Step 1] Creating data subsets for rapid demonstration...")

    # We assume the metadata files exist as per the problem description
    # We take 50 breaths for train, 20 for val, 20 for test to keep it very fast
    subset_train_path = os.path.join(demo_input_dir, "train_subset.csv")
    subset_val_path = os.path.join(demo_input_dir, "val_subset.csv")
    subset_test_path = os.path.join(demo_input_dir, "test_subset.csv")

    create_subset_data(Config.TRAIN_CSV, subset_train_path, num_breaths=50)
    create_subset_data(Config.VAL_CSV, subset_val_path, num_breaths=20)
    create_subset_data(Config.TEST_CSV, subset_test_path, num_breaths=20)

    # Copy sample submission and truncate it to match the test subset size
    # Test subset has 20 breaths * 80 steps = 1600 rows
    sample_sub_orig = pd.read_csv(os.path.join("./input", "sample_submission.csv"))
    sample_sub_subset = sample_sub_orig.head(20 * 80).copy()
    # We need to ensure the IDs match the test subset.
    # The test subset was taken from the top of test.csv, so IDs should align with top of sample_submission.
    subset_sample_sub_path = os.path.join(demo_input_dir, "sample_submission.csv")
    sample_sub_subset.to_csv(subset_sample_sub_path, index=False)

    # 3. Monkeypatch Config to use Demo Paths
    print("\n[Step 2] Configuring library to use demo paths...")
    Config.INPUT_DIR = demo_input_dir
    Config.TRAIN_CSV = subset_train_path
    Config.VAL_CSV = subset_val_path
    Config.TEST_CSV = subset_test_path
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    # Adjust Hyperparameters for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1
    Config.LSTM_HIDDEN = 64  # Reduce model size for speed
    Config.TCN_CHANNELS = [32, 64]  # Reduce model depth

    Config.print_config()

    # 4. Verify Data Pipeline
    print("\n[Step 3] Verifying Data Processing Pipeline...")
    # Force reprocessing to ensure our subsets are used
    data_dict = prepare_datasets(load_cached_data=False)

    # Assertions for Data Shapes
    # Shape should be (Num_Breaths, Seq_Len, Num_Features)
    assert data_dict["train_x"].ndim == 3, "Train X should be 3D"
    assert (
        data_dict["train_x"].shape[1] == Config.SEQ_LEN
    ), f"Sequence length must be {Config.SEQ_LEN}"
    assert (
        data_dict["train_x"].shape[2] == Config.INPUT_DIM
    ), f"Feature dim must be {Config.INPUT_DIM}"
    assert data_dict["train_y"].shape == (50, 80), "Train Y shape mismatch"
    assert data_dict["train_u_out"].shape == (50, 80), "Train u_out shape mismatch"

    print("Data Pipeline verification passed. Shapes are correct.")

    # 5. Verify Model and Loss Logic
    print("\n[Step 4] Verifying Model and Loss Logic...")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = PCANet(Config).to(device)
    model.eval()

    # Create a dummy batch
    batch_size = 2
    dummy_input = torch.randn(batch_size, Config.SEQ_LEN, Config.INPUT_DIM).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check Output Shape: (Batch, Seq_Len)
    assert output.shape == (
        batch_size,
        Config.SEQ_LEN,
    ), f"Model output shape mismatch: {output.shape}"
    print("Model forward pass successful.")

    # Verify MaskedMAELoss
    criterion = MaskedMAELoss()

    # Case 1: Perfect prediction
    pred = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    target = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    u_out = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # All inspiratory
    loss = criterion(pred, target, u_out)
    assert loss.item() == 0.0, "Loss should be 0 for perfect prediction"

    # Case 2: Error only in expiratory phase (u_out=1) -> Should be ignored
    pred = torch.tensor([[100.0]])  # Huge error
    target = torch.tensor([[0.0]])
    u_out = torch.tensor([[1.0]])  # Expiratory
    loss = criterion(pred, target, u_out)
    assert loss.item() == 0.0, "Loss should be 0 when u_out=1 (masked)"

    # Case 3: Error in inspiratory phase
    pred = torch.tensor([[12.0]])
    target = torch.tensor([[10.0]])
    u_out = torch.tensor([[0.0]])
    loss = criterion(pred, target, u_out)
    assert abs(loss.item() - 2.0) < 1e-6, "Loss should be |12-10| = 2"

    print("MaskedMAELoss logic verified.")

    # 6. Verify Training Loop
    print("\n[Step 5] Executing Training Loop (1 Epoch)...")

    # We call train_model. It will reload data using get_dataloaders.
    # We set load_cached_data=True because we just generated the cache in Step 3 (prepare_datasets)
    # and we want to use that cache which corresponds to our subsets.
    train_model(
        epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 7. Verify Submission
    print("\n[Step 6] Verifying Submission Generation...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check format
    expected_cols = ["id", "pressure"]
    if list(sub_df.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
        )

    # Check length (Test subset was 20 breaths * 80 steps = 1600)
    expected_len = 20 * 80
    if len(sub_df) != expected_len:
        raise ValueError(
            f"Submission length mismatch. Expected {expected_len}, got {len(sub_df)}"
        )

    # Check values are not all zero (model learned something or initialized randomly non-zero)
    if sub_df["pressure"].abs().sum() == 0:
        print(
            "Warning: All predictions are exactly zero. This might indicate an issue, but technically possible with init."
        )

    print(f"Submission verified. File saved at {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
