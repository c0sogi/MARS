import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders, compute_features
from library.model import DP_GI_BiLSTM
from library.train import train_epoch, validate, predict, get_u_out_metadata


def create_mini_dataset():
    """
    Creates a small subset of the data for demonstration purposes.
    Saves mini csvs and metadata to the working directory.
    """
    print("Creating mini dataset for demonstration...")

    # Define paths
    working_dir = "./working/demo_data"
    os.makedirs(working_dir, exist_ok=True)

    mini_train_path = os.path.join(working_dir, "mini_train.csv")
    mini_test_path = os.path.join(working_dir, "mini_test.csv")

    # Read a small chunk of data (enough for ~10 breaths, 80 steps per breath)
    # 10 breaths * 80 steps = 800 rows. Let's grab 2000 to be safe.
    df_train_full = pd.read_csv(Config.TRAIN_CSV, nrows=2000)
    df_test_full = pd.read_csv(Config.TEST_CSV, nrows=2000)

    # Get unique breath IDs
    train_breath_ids = df_train_full["breath_id"].unique()
    test_breath_ids = df_test_full["breath_id"].unique()

    # Select 4 breaths for train, 2 for val
    if len(train_breath_ids) < 6:
        raise ValueError("Not enough breaths in the first 2000 rows of train.csv")

    train_ids = train_breath_ids[:4]
    val_ids = train_breath_ids[4:6]

    # Select 2 breaths for test
    test_ids_subset = test_breath_ids[:2]

    # Filter Dataframes
    df_mini_train = df_train_full[
        df_train_full["breath_id"].isin(np.concatenate([train_ids, val_ids]))
    ].copy()
    df_mini_test = df_test_full[df_test_full["breath_id"].isin(test_ids_subset)].copy()

    # Save Mini CSVs
    df_mini_train.to_csv(mini_train_path, index=False)
    df_mini_test.to_csv(mini_test_path, index=False)

    # Create Metadata
    # Train Meta
    train_meta = df_mini_train[df_mini_train["breath_id"].isin(train_ids)][
        ["id", "breath_id", "pressure"]
    ].copy()
    train_meta["source_file"] = (
        "mini_train.csv"  # Relative to input dir usually, but we will patch logic or path
    )

    # Val Meta
    val_meta = df_mini_train[df_mini_train["breath_id"].isin(val_ids)][
        ["id", "breath_id", "pressure"]
    ].copy()
    val_meta["source_file"] = "mini_train.csv"

    # Test Meta
    test_meta = df_mini_test[["id", "breath_id"]].copy()
    test_meta["source_file"] = "mini_test.csv"

    # Save Metadata
    train_meta_path = os.path.join(working_dir, "train_meta.csv")
    val_meta_path = os.path.join(working_dir, "val_meta.csv")
    test_meta_path = os.path.join(working_dir, "test_meta.csv")

    train_meta.to_csv(train_meta_path, index=False)
    val_meta.to_csv(val_meta_path, index=False)
    test_meta.to_csv(test_meta_path, index=False)

    return {
        "train_csv": mini_train_path,
        "test_csv": mini_test_path,
        "train_meta": train_meta_path,
        "val_meta": val_meta_path,
        "test_meta": test_meta_path,
        "cache_dir": os.path.join(working_dir, "cache"),
    }


def patch_config(paths):
    """
    Modifies Config singleton to point to mini datasets and reduce runtime.
    """
    print("Patching Configuration...")
    Config.TRAIN_CSV = paths["train_csv"]
    Config.TEST_CSV = paths["test_csv"]

    Config.TRAIN_META = paths["train_meta"]
    Config.VAL_META = paths["val_meta"]
    Config.TEST_META = paths["test_meta"]

    Config.CACHE_DIR = paths["cache_dir"]
    Config.WORKING_DIR = os.path.dirname(paths["cache_dir"])
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Runtime optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.LSTM_HIDDEN = 64  # Reduce model size for speed
    Config.LSTM_LAYERS = 2

    # Ensure cache dir exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)


def verify_data_loading():
    """
    Verifies that dataloaders are constructed correctly and yield valid shapes.
    """
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Check Config update
    print(f"Dynamic Input Dimension: {Config.INPUT_DIM}")
    assert (
        Config.INPUT_DIM > 0
    ), "Input dimension should be positive after feature engineering."

    # Check Train Loader
    X_batch, y_batch = next(iter(train_loader))
    print(f"Train Batch Shape - X: {X_batch.shape}, y: {y_batch.shape}")

    # Assertions
    # Shape: (Batch_Size, Seq_Len, Features)
    assert X_batch.shape[0] == Config.BATCH_SIZE
    assert X_batch.shape[1] == 80  # Standard sequence length
    assert X_batch.shape[2] == Config.INPUT_DIM
    assert y_batch.shape == (Config.BATCH_SIZE, 80)

    print("Data Loading Verified.")
    return train_loader, val_loader, test_loader, test_ids


def verify_model_forward(device):
    """
    Verifies model instantiation and forward pass.
    """
    print("\n--- Verifying Model Architecture ---")
    model = DP_GI_BiLSTM(Config).to(device)

    # Create dummy input
    dummy_input = torch.randn(Config.BATCH_SIZE, 80, Config.INPUT_DIM).to(device)
    output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (Config.BATCH_SIZE, 80), "Output shape mismatch."
    assert not torch.isnan(output).any(), "Model produced NaNs."

    print("Model Architecture Verified.")
    return model


def verify_training_loop(model, train_loader, val_loader, device):
    """
    Verifies the training step and validation step.
    """
    print("\n--- Verifying Training Loop ---")

    # Get u_out metadata for loss calculation
    u_out_idx, u_out_mean, u_out_std = get_u_out_metadata(Config.CACHE_DIR)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch
    loss = train_epoch(
        model, train_loader, optimizer, device, u_out_idx, u_out_mean, u_out_std
    )
    print(f"Epoch Loss: {loss:.4f}")

    assert loss > 0, "Loss should be positive."
    assert not np.isnan(loss), "Loss is NaN."

    # Run validation
    mae = validate(model, val_loader, device, u_out_idx, u_out_mean, u_out_std)
    print(f"Validation MAE: {mae:.4f}")

    assert mae >= 0, "MAE should be non-negative."

    # Save model for inference test
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
    print("Training Loop Verified.")


def verify_inference(model, test_loader, test_ids, device):
    """
    Verifies prediction generation and submission file creation.
    """
    print("\n--- Verifying Inference ---")

    # Load weights (simulating inference script flow)
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))

    preds = predict(model, test_loader, device)

    print(f"Predictions generated: {len(preds)}")
    print(f"Test IDs count: {len(test_ids)}")

    assert len(preds) == len(test_ids), "Mismatch between predictions and test IDs."

    # Create submission
    submission_df = pd.DataFrame({"id": test_ids, "pressure": preds})
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    # Check content
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert df_check.shape[0] == len(test_ids)
    assert df_check.shape[1] == 2

    print("Inference Verified.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # 2. Prepare Data
    paths = create_mini_dataset()
    patch_config(paths)

    # 3. Verify Components
    train_loader, val_loader, test_loader, test_ids = verify_data_loading()

    model = verify_model_forward(device)

    verify_training_loop(model, train_loader, val_loader, device)

    verify_inference(model, test_loader, test_ids, device)

    print("\nAll verification steps completed successfully.")
