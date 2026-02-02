import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import rle_encoding, calculate_f05, set_seed
from library.data import get_specialist_datasets, InkDataset
from library.model import SpecialistSegFormer
from library.trainer import train_specialist
from library.inference import create_submission


def run_demo():
    print("=== Starting Vesuvius Ink Detection Library Demo ===")

    # --- 1. Configuration Overrides for Demo Speed ---
    # We override Config parameters to ensure the script runs quickly (within minutes)
    # and uses a separate working directory to avoid conflicts.
    DEMO_DIR = "./working/demo_execution"
    DEMO_META_DIR = os.path.join(DEMO_DIR, "metadata")

    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(DEMO_META_DIR, exist_ok=True)

    print(f"Setting up configuration. Working Dir: {DEMO_DIR}")

    # Monkey-patch Config
    Config.WORKING_DIR = DEMO_DIR
    Config.METADATA_DIR = DEMO_META_DIR
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.VALID_THRESHOLD = 0.0  # Force save model regardless of score
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set reproducibility
    set_seed(42)

    # --- 2. Verify Utility Functions ---
    print("\n--- Verifying Utility Functions ---")

    # Test RLE Encoding
    # Create a simple 4x4 mask:
    # 0 1 1 0
    # 0 0 0 0
    # 1 0 0 0
    # 0 0 0 0
    # Flattened: 0, 1, 1, 0, 0, 0, 0, 0, 1, 0...
    # Indices (1-based): 2, 3 have 1s. Then index 9 has 1.
    # RLE should be: "2 2 9 1"
    dummy_mask = np.zeros((4, 4), dtype=np.uint8)
    dummy_mask[0, 1:3] = 1
    dummy_mask[2, 0] = 1

    rle_str = rle_encoding(dummy_mask)
    print(f"RLE Output: {rle_str}")
    assert (
        rle_str == "2 2 9 1"
    ), f"RLE Encoding failed. Expected '2 2 9 1', got '{rle_str}'"

    # Test F0.5 Score
    # Perfect match
    y_true = torch.tensor([1, 0, 1, 0])
    y_pred = torch.tensor([0.9, 0.1, 0.8, 0.2])  # Probabilities
    score = calculate_f05(y_true, y_pred, threshold=0.5)
    print(f"F0.5 Score (Perfect): {score}")
    assert np.isclose(score, 1.0), "F0.5 calculation failed for perfect match"

    # --- 3. Prepare Data Subset ---
    print("\n--- Preparing Data Subset ---")

    # We read the actual metadata and sample a few rows to create a mini-dataset
    # This avoids processing the entire 400+ patch dataset.
    original_train_csv = "./metadata/train.csv"
    original_val_csv = "./metadata/validation.csv"
    original_test_csv = "./metadata/test.csv"

    if not os.path.exists(original_train_csv):
        raise FileNotFoundError(
            "Original metadata not found. Please ensure ./metadata exists."
        )

    # Load and sample
    df_train = pd.read_csv(original_train_csv)
    df_val = pd.read_csv(original_val_csv)
    df_test = pd.read_csv(original_test_csv)

    # Take top 8 patches from train (Fragment 1 usually)
    df_train_sub = df_train[df_train["fragment_id"] == 1].head(8)
    # Take top 4 patches from val
    df_val_sub = df_val[df_val["fragment_id"] == 1].head(4)
    # Use the test fragment 'a'
    df_test_sub = df_test.head(1)

    # Save to demo metadata directory
    df_train_sub.to_csv(os.path.join(DEMO_META_DIR, "train.csv"), index=False)
    df_val_sub.to_csv(os.path.join(DEMO_META_DIR, "validation.csv"), index=False)
    df_test_sub.to_csv(os.path.join(DEMO_META_DIR, "test.csv"), index=False)

    print(
        f"Created subset metadata: {len(df_train_sub)} train, {len(df_val_sub)} val, {len(df_test_sub)} test."
    )

    # --- 4. Data Processing & Loading ---
    print("\n--- Processing Data & Creating Loaders ---")

    # Generate data for Specialist 'A' (High Z-range)
    # load_cached_data=False forces generation from our new subset metadata
    train_ds, val_ds = get_specialist_datasets("A", load_cached_data=False)

    print(f"Train Dataset Length: {len(train_ds)}")
    print(f"Val Dataset Length: {len(val_ds)}")

    assert len(train_ds) == 8, "Train dataset size mismatch"

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Check a single batch
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Should be (B, 3, 512, 512)
    print(f"Batch Label Shape: {labels.shape}")  # Should be (B, 1, 512, 512)

    assert images.shape == (Config.BATCH_SIZE, 3, 512, 512)
    assert labels.shape == (Config.BATCH_SIZE, 1, 512, 512)

    # --- 5. Model Initialization ---
    print("\n--- Initializing SpecialistSegFormer Model ---")

    model = SpecialistSegFormer()
    device = torch.device(Config.DEVICE)
    model.to(device)

    # Dummy forward pass
    with torch.no_grad():
        dummy_input = images.to(device)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
        512,
        512,
    ), "Model output shape mismatch"

    # --- 6. Training Loop Demo ---
    print("\n--- Running Training Loop (1 Epoch) ---")

    # Train Specialist A
    best_score = train_specialist(model, train_loader, val_loader, specialist_key="A")

    print(f"Training complete. Best Score: {best_score}")

    # Verify model checkpoint exists
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_A_best.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    # For the purpose of the demo inference, we will copy model A to B and C
    # so the ensemble inference doesn't complain about missing models (or use random weights).
    shutil.copy(checkpoint_path, os.path.join(Config.WORKING_DIR, "model_B_best.pth"))
    shutil.copy(checkpoint_path, os.path.join(Config.WORKING_DIR, "model_C_best.pth"))
    print("Copied Model A to B and C for ensemble demo.")

    # --- 7. Inference Pipeline Demo ---
    print("\n--- Running Inference Pipeline ---")

    # This will read test.csv, load the models, predict, and save submission.csv
    create_submission()

    submission_file = os.path.join(Config.WORKING_DIR, Config.SUBMISSION_PATH)
    if os.path.exists(Config.SUBMISSION_PATH):
        # The library saves to Config.SUBMISSION_PATH which defaults to "submission.csv" in current dir
        # But we might want to check where it actually saved based on CWD or Config path.
        # Config.SUBMISSION_PATH is just a filename "submission.csv".
        # The library code does: df.to_csv(output_path) where output_path = Config.SUBMISSION_PATH
        pass

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    assert len(df_sub) == len(
        df_test_sub
    ), "Submission rows do not match test fragments"
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns invalid"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
