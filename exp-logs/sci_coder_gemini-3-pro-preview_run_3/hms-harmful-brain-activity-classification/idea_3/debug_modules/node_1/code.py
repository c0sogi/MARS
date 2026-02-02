import os
import shutil
import warnings
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloader
from library.model import EEGNet
from library.train import train
from library.inference import predict

# ==========================================
# Setup & Configuration Overrides
# ==========================================
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Override Config for a fast demonstration run
Config.EPOCHS = 1
Config.BATCH_SIZE = 4
Config.WORKING_DIR = "./working/demo_run"
Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.SUBMISSION_DIR = "./working/demo_submission"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Clean up previous runs if they exist
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
os.makedirs(Config.WORKING_DIR, exist_ok=True)

if os.path.exists(Config.SUBMISSION_DIR):
    shutil.rmtree(Config.SUBMISSION_DIR)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Define subset size for speed
DEMO_SUBSET_SIZE = 50

if __name__ == "__main__":
    print(">>> Starting EEG Activity Detection Pipeline Demo")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # ==========================================
    # 1. Data Pipeline Verification
    # ==========================================
    print("\n[Step 1] Verifying Data Loading & Preprocessing...")

    # Initialize DataLoader with a small subset
    train_loader = get_dataloader(
        mode="train",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing from scratch for demo
        debug_subset=DEMO_SUBSET_SIZE,
    )

    # Fetch a single batch
    try:
        images, targets = next(iter(train_loader))
        print(
            f"  Batch Loaded. Images Shape: {images.shape}, Targets Shape: {targets.shape}"
        )
    except StopIteration:
        raise AssertionError("DataLoader returned no data!")

    # Assertions
    # Expected Image Shape: (Batch, 19 Channels, 512 Height, 512 Width)
    assert images.dim() == 4, f"Expected 4D input tensor, got {images.dim()}D"
    assert images.shape[1] == 19, f"Expected 19 EEG channels, got {images.shape[1]}"
    assert (
        images.shape[2:] == Config.IMG_SIZE
    ), f"Expected image size {Config.IMG_SIZE}, got {images.shape[2:]}"

    # Expected Target Shape: (Batch, 6 Classes)
    assert targets.shape[1] == 6, f"Expected 6 target classes, got {targets.shape[1]}"

    # Verify Targets are Probabilities (Sum close to 1)
    target_sums = targets.sum(dim=1)
    assert torch.allclose(
        target_sums, torch.ones_like(target_sums), atol=1e-5
    ), "Target probabilities do not sum to 1"

    print("  Data pipeline verification passed.")

    # ==========================================
    # 2. Model Architecture Verification
    # ==========================================
    print("\n[Step 2] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(
        pretrained=False
    )  # Use false to avoid download overhead/errors in demo
    model.to(device)
    model.eval()

    # Move batch to device
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    print(f"  Forward Pass Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 6)}, got {outputs.shape}"

    # Verify Softmax Output (Sum to 1, all positive)
    output_sums = outputs.sum(dim=1)
    assert torch.allclose(
        output_sums, torch.ones_like(output_sums), atol=1e-5
    ), "Model outputs do not sum to 1 (Softmax missing?)"
    assert (outputs >= 0).all(), "Model outputs contain negative values"

    print("  Model architecture verification passed.")

    # ==========================================
    # 3. Training Loop Execution
    # ==========================================
    print("\n[Step 3] Running Training Loop (Fast Mode)...")
    print(f"  Training for {Config.EPOCHS} epoch(s) on {DEMO_SUBSET_SIZE} samples.")

    # Run training
    # Note: This will save the best model to Config.MODEL_PATH
    train(debug_subset_size=DEMO_SUBSET_SIZE)

    # Verify model artifact creation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed to generate model file at {Config.MODEL_PATH}"
        )

    print(f"  Training complete. Model saved to {Config.MODEL_PATH}")

    # ==========================================
    # 4. Inference Execution
    # ==========================================
    print("\n[Step 4] Running Inference...")

    # Run inference on a subset of test data
    # Note: predict() handles loading the model from Config.MODEL_PATH
    submission_df = predict(debug_subset_size=DEMO_SUBSET_SIZE)

    print("  Inference complete.")

    # ==========================================
    # 5. Submission Validation
    # ==========================================
    print("\n[Step 5] Validating Submission File...")

    # Check file existence
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Reload to verify format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Columns
    expected_cols = ["eeg_id"] + [
        c.replace("_prob", "_vote") for c in Config.TARGET_COLS
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch.\nExpected: {expected_cols}\nGot: {list(df_sub.columns)}"

    # Check Row Count
    # We used debug_subset_size=DEMO_SUBSET_SIZE for prediction
    assert (
        len(df_sub) == DEMO_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {DEMO_SUBSET_SIZE}, got {len(df_sub)}"

    # Check Probability Sums
    vote_cols = expected_cols[1:]
    row_sums = df_sub[vote_cols].sum(axis=1)
    # Allow small floating point error
    invalid_sums = row_sums[~np.isclose(row_sums, 1.0, atol=1e-5)]

    if not invalid_sums.empty:
        print(f"  Invalid Sums found:\n{invalid_sums.head()}")
        raise AssertionError("Submission rows do not sum to 1.0")

    print("  Submission file is valid.")
    print("\n>>> Demo Completed Successfully.")
