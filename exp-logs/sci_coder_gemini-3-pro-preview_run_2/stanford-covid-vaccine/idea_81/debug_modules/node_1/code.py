import os
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.model import AHDRNModel
from library.engine import train_model, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_mini_dataset(source_path, dest_path, num_samples=20):
    """Creates a smaller version of the dataset for demonstration purposes."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file {source_path} does not exist.")

    df = pd.read_csv(source_path)
    # Take a subset
    df_mini = df.head(num_samples).copy()

    # Ensure destination directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df_mini.to_csv(dest_path, index=False)
    print(f"Created mini dataset at {dest_path} with {len(df_mini)} samples.")
    return len(df_mini)


def main():
    print("=== Starting Demonstration Script ===\n")

    # 1. Setup Directories and Patch Config
    # We patch the Config class directly to affect all downstream modules
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print("Step 1: Configuring environment and creating mini-datasets...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(demo_dir, "metadata", "train.csv")
    mini_val_path = os.path.join(demo_dir, "metadata", "val.csv")
    mini_test_path = os.path.join(demo_dir, "metadata", "test.csv")

    # Create mini datasets
    n_train = create_mini_dataset(Config.TRAIN_CSV, mini_train_path, num_samples=32)
    n_val = create_mini_dataset(Config.VAL_CSV, mini_val_path, num_samples=16)
    n_test = create_mini_dataset(Config.TEST_CSV, mini_test_path, num_samples=10)

    # Patch Config
    Config.TRAIN_CSV = mini_train_path
    Config.VAL_CSV = mini_val_path
    Config.TEST_CSV = mini_test_path

    Config.WORKING_DIR = os.path.join(demo_dir, "cache")
    Config.CACHE_TRAIN = os.path.join(Config.WORKING_DIR, "train_data_hc_sdrn_v1.npz")
    Config.CACHE_VAL = os.path.join(Config.WORKING_DIR, "val_data_hc_sdrn_v1.npz")
    Config.CACHE_TEST = os.path.join(Config.WORKING_DIR, "test_data_hc_sdrn_v1.npz")

    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission", "submission.csv")

    # Speed up training for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.PATIENCE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    set_seed(Config.SEED)

    # 2. Verify Data Loading
    print("\nStep 2: Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch
    inputs, partner_indices, targets = next(iter(train_loader))

    print(
        f"Batch shapes -> Inputs: {inputs.shape}, Partner: {partner_indices.shape}, Targets: {targets.shape}"
    )

    # Assertions
    # Inputs: (B, 107, 18) -> 18 channels as defined in model.py (4+3+7+4)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        18,
    ), f"Expected input shape ({Config.BATCH_SIZE}, 107, 18), got {inputs.shape}"

    # Partner indices: (B, 107)
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Expected partner shape ({Config.BATCH_SIZE}, 107), got {partner_indices.shape}"

    # Targets: (B, 107, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Expected target shape ({Config.BATCH_SIZE}, 107, 5), got {targets.shape}"

    print("Data loading verification passed.")

    # 3. Verify Model Architecture
    print("\nStep 3: Verifying Model Architecture...")
    device = torch.device("cpu")  # Use CPU for simple logic check
    model = AHDRNModel().to(device)

    # Forward pass (Training mode returns tuple: y_2, y_1)
    model.train()
    y_2, y_1 = model(inputs, partner_indices)

    print(f"Model Output Shapes (Train) -> Main: {y_2.shape}, Aux: {y_1.shape}")

    assert y_2.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Main output shape mismatch"
    assert y_1.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Aux output shape mismatch"

    # Forward pass (Eval mode returns tensor: y_2)
    model.eval()
    with torch.no_grad():
        y_eval = model(inputs, partner_indices)

    print(f"Model Output Shape (Eval) -> {y_eval.shape}")
    assert y_eval.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), "Eval output shape mismatch"

    print("Model architecture verification passed.")

    # 4. Verify Loss Calculation
    print("\nStep 4: Verifying Loss Calculation...")
    criterion = MCRMSELoss()

    # Calculate loss
    loss = criterion(y_2, targets)
    print(f"Calculated Loss: {loss.item()}")

    assert isinstance(loss.item(), float), "Loss should be a float"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Loss function verification passed.")

    # 5. Run Training Loop
    print("\nStep 5: Running Training Loop (Mini-Train)...")
    # This calls engine.train_model which uses the patched Config
    best_metric = train_model()

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    print(f"Training finished. Best MCRMSE: {best_metric}")

    # 6. Generate Submission
    print("\nStep 6: Generating Submission...")
    generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Expected rows: n_test samples * 107 positions
    expected_rows = n_test * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    print("Submission verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
