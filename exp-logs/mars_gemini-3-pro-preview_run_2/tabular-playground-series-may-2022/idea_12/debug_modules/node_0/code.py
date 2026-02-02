import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import GlobalContextTransformerResFunnel
from library.train import run_training, generate_submission


def run_demo():
    print("=== Starting Manufacturing Control Task Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Override Config for a quick demonstration run
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update dependent paths to use the demo directory
    Config.PROCESSED_DATA_PATH = os.path.join(Config.WORKING_DIR, "processed_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Set hyperparams for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.DEBUG = True  # Limits dataset to 5000 samples
    Config.NUM_WORKERS = 2  # Adjust based on vCPU availability

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Clear any existing cache to prove data processing works
    if os.path.exists(Config.PROCESSED_DATA_PATH):
        os.remove(Config.PROCESSED_DATA_PATH)

    # Load data (this will trigger preprocess_data internally)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    cont = batch["continuous"]
    seq = batch["sequence"]
    target = batch["target"]

    print(f"    Batch keys: {list(batch.keys())}")
    print(f"    Continuous features shape: {cont.shape}")
    print(f"    Sequence features shape:   {seq.shape}")
    print(f"    Target shape:              {target.shape}")

    # Logic Verification
    assert cont.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CONTINUOUS_FEATURES,
    ), f"Expected continuous shape ({Config.BATCH_SIZE}, {Config.NUM_CONTINUOUS_FEATURES}), got {cont.shape}"
    assert seq.shape == (
        Config.BATCH_SIZE,
        Config.CHAR_SEQ_LEN,
    ), f"Expected sequence shape ({Config.BATCH_SIZE}, {Config.CHAR_SEQ_LEN}), got {seq.shape}"
    assert target.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    # --------------------------------------------------------------------------
    # 3. Model Architecture Demonstration
    # --------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Instantiation & Forward Pass...")

    model = GlobalContextTransformerResFunnel().to(device)

    # Prepare inputs
    cont = cont.to(device)
    seq = seq.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(cont, seq)

    print(f"    Model Output Shape: {output.shape}")

    # Logic Verification
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {output.shape}"
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Model probabilities are out of bounds [0, 1]"

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[4] Demonstrating Training Loop...")

    # run_training encapsulates the loop, validation, and saving
    # We use debug=True to ensure it uses the subset and runs quickly
    best_auc = run_training(epochs=Config.EPOCHS, debug=True)

    print(f"    Training finished. Best Validation AUC: {best_auc:.4f}")

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"

    # --------------------------------------------------------------------------
    # 5. Inference & Submission Demonstration
    # --------------------------------------------------------------------------
    print("\n[5] Demonstrating Inference and Submission Generation...")

    generate_submission(debug=True)

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")

    # --------------------------------------------------------------------------
    # 6. Submission Validation
    # --------------------------------------------------------------------------
    print("\n[6] Validating Submission Content...")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission DataFrame Shape: {df_sub.shape}")
    print(f"    Head:\n{df_sub.head()}")

    # Verify Columns
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission missing required columns 'id' and 'target'"

    # Verify Value Range
    assert (
        df_sub["target"].min() >= 0 and df_sub["target"].max() <= 1
    ), "Prediction values out of valid probability range [0, 1]"

    # Verify ID Consistency
    # Since we are in debug mode, the test set is truncated.
    # We must ensure the submission contains exactly the IDs from the test loader.
    expected_ids = []
    for b in test_loader:
        expected_ids.append(b["id"].numpy())
    expected_ids = np.concatenate(expected_ids)

    sub_ids = df_sub["id"].values

    # Sort for comparison
    expected_ids.sort()
    sub_ids.sort()

    np.testing.assert_array_equal(
        sub_ids, expected_ids, err_msg="Submission IDs do not match the Test Set IDs"
    )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
