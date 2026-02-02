import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import VentilatorDataset, add_features
from library.model import PCDRHNet
from library.loss import MaskedL1Loss
from library.train import load_data_and_create_loaders, train_epoch, validate, predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_demo_subsets():
    """
    Creates small subsets of the metadata files to speed up the demonstration.
    Returns the paths to these new subset files.
    """
    print("\n[Demo] Creating data subsets...")

    # Define subset paths
    subset_dir = "./working/demo_input"
    os.makedirs(subset_dir, exist_ok=True)

    train_sub_path = os.path.join(subset_dir, "train_subset.csv")
    val_sub_path = os.path.join(subset_dir, "val_subset.csv")
    test_sub_path = os.path.join(subset_dir, "test_subset.csv")

    # Helper to read first N breaths (N * 80 rows)
    def save_subset(src_path, dest_path, n_breaths=20):
        # Read a chunk large enough to cover n_breaths
        # Assuming 80 rows per breath, read (n_breaths + 5) * 80 to be safe
        chunk_size = (n_breaths + 5) * 80
        if os.path.exists(src_path):
            df = pd.read_csv(src_path, nrows=chunk_size)

            # Filter exactly n_breaths
            unique_breaths = df["breath_id"].unique()[:n_breaths]
            df_subset = df[df["breath_id"].isin(unique_breaths)]

            df_subset.to_csv(dest_path, index=False)
            print(
                f"  Created {dest_path} with {len(df_subset)} rows ({n_breaths} breaths)."
            )
        else:
            raise FileNotFoundError(f"Source file {src_path} not found.")

    # Create subsets
    save_subset(Config.TRAIN_PATH, train_sub_path, n_breaths=20)
    save_subset(Config.VAL_PATH, val_sub_path, n_breaths=10)
    save_subset(Config.TEST_PATH, test_sub_path, n_breaths=10)

    return train_sub_path, val_sub_path, test_sub_path


def verify_loss_logic():
    """
    Verifies that MaskedL1Loss correctly ignores errors during the expiratory phase (u_out=1).
    """
    print("\n[Demo] Verifying MaskedL1Loss logic...")
    criterion = MaskedL1Loss()

    # Setup: 2 samples
    # Sample 1: u_out=0 (Inspiratory) -> Error should count
    # Sample 2: u_out=1 (Expiratory) -> Error should be ignored

    pred = torch.tensor([10.0, 10.0])
    target = torch.tensor([12.0, 20.0])  # Errors: 2.0 and 10.0
    u_out = torch.tensor([0.0, 1.0])  # Mask: Keep 1st, Drop 2nd

    loss = criterion(pred, target, u_out)

    # Expected calculation:
    # Abs Error: [2.0, 10.0]
    # Mask: [1.0, 0.0]
    # Masked Error: [2.0, 0.0]
    # Mean: Sum(2.0) / Sum(Mask=1) = 2.0 / 1.0 = 2.0

    expected_loss = 2.0
    assert (
        abs(loss.item() - expected_loss) < 1e-6
    ), f"Loss logic failed. Expected {expected_loss}, got {loss.item()}"

    print("  MaskedL1Loss logic verified: Expiratory phase error correctly ignored.")


def run_demo():
    # 1. Setup Environment
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Demo] Running on device: {device}")

    # 2. Override Config for Demo
    # We point to a new working directory to ensure fresh cache generation
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Create subsets and override paths
    train_sub, val_sub, test_sub = create_demo_subsets()
    Config.TRAIN_PATH = train_sub
    Config.VAL_PATH = val_sub
    Config.TEST_PATH = test_sub

    # Override Hyperparameters for speed
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.TCN_LAYERS = 2  # Reduce model depth for speed
    Config.LSTM_LAYERS = 1

    # 3. Verify Loss Function
    verify_loss_logic()

    # 4. Data Loading & Feature Engineering
    print("\n[Demo] Loading data and generating features...")
    # This function handles feature engineering, caching, scaling, and reshaping
    train_loader, val_loader, test_loader, test_df = load_data_and_create_loaders()

    # Verify Data Shapes
    print("[Demo] Verifying DataLoader shapes...")
    x_a, x_b, y = next(iter(train_loader))

    # Expected: (Batch, 80, Features)
    print(f"  Stream A (Features) Shape: {x_a.shape}")
    print(f"  Stream B (Mask) Shape:     {x_b.shape}")
    print(f"  Target Shape:              {y.shape}")

    assert x_a.shape[0] == Config.BATCH_SIZE
    assert x_a.shape[1] == 80
    assert x_b.shape[1] == 80
    assert y.shape[1] == 80
    assert x_b.shape[2] == 1  # u_out is single feature in Stream B

    # 5. Model Initialization & Verification
    print("\n[Demo] Initializing Model...")
    input_dim = x_a.shape[2]
    model = PCDRHNet(input_dim).to(device)

    # Verify Forward Pass
    print("[Demo] Verifying Model Forward Pass...")
    with torch.no_grad():
        dummy_input = x_a.to(device)
        output = model(dummy_input)

    print(f"  Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        80,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 80)}, got {output.shape}"

    # 6. Training Loop
    print("\n[Demo] Starting Training Loop...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = MaskedL1Loss()

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(
            f"  Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Basic assertion to ensure model is learning or at least not exploding
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    # 7. Inference & Submission
    print("\n[Demo] Running Inference...")
    predictions = predict(model, test_loader, device)

    # Verify Prediction Shape
    expected_rows = len(test_df)
    predictions_flat = predictions.flatten()
    print(
        f"  Predictions shape: {predictions.shape} -> Flattened: {predictions_flat.shape}"
    )

    assert (
        len(predictions_flat) == expected_rows
    ), f"Prediction count mismatch. Expected {expected_rows}, got {len(predictions_flat)}"

    # Generate Submission
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission = pd.DataFrame(
        {Config.ID_COL: test_df[Config.ID_COL], Config.TARGET_COL: predictions_flat}
    )
    submission.to_csv(submission_path, index=False)
    print(f"  Submission saved to {submission_path}")

    # Final check of submission file
    saved_df = pd.read_csv(submission_path)
    assert saved_df.shape == (expected_rows, 2), "Submission file has incorrect shape"
    assert list(saved_df.columns) == ["id", "pressure"], "Submission columns mismatch"

    print("\n[Demo] Demonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
