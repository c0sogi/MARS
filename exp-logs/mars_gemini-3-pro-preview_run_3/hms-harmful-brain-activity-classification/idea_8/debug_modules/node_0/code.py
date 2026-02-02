import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloader
from library.model import AttentiveDualScaleNetwork
from library.train import run_training
from library.inference import predict


# ==========================================
# Setup & Configuration
# ==========================================
def setup_environment():
    """
    Sets up the environment for the demo run.
    Suppresses warnings and patches the Config class for speed.
    """
    # Suppress warnings
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Set seeds
    seed_everything(Config.SEED)

    print("Setting up demonstration environment...")

    # Define a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # --- Monkey-Patch Config for Speed ---
    # We modify the Config class attributes directly to force the library
    # to use our settings without modifying the source file.
    Config.WORKING_DIR = demo_dir
    Config.OUTPUT_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")

    # Reduce compute requirements
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Point to debug metadata files (will be created shortly)
    Config.TRAIN_CSV = os.path.join(demo_dir, "train_debug.csv")
    Config.VAL_CSV = os.path.join(demo_dir, "val_debug.csv")
    Config.TEST_CSV = os.path.join(demo_dir, "test_debug.csv")

    print(f"Working directory set to: {Config.WORKING_DIR}")


def create_debug_data():
    """
    Reads the original metadata, samples a tiny subset, and saves it
    to the working directory. This ensures the dataloaders only process
    a small amount of data.
    """
    print("\nCreating debug datasets...")

    # 1. Train Data
    orig_train_path = "./metadata/train.csv"
    if not os.path.exists(orig_train_path):
        raise FileNotFoundError(
            f"Original train metadata not found at {orig_train_path}"
        )

    df_train = pd.read_csv(orig_train_path)
    # Take 16 samples for training (4 batches of 4)
    df_train_debug = df_train.head(16).copy()
    df_train_debug.to_csv(Config.TRAIN_CSV, index=False)
    print(f"Created {Config.TRAIN_CSV} with {len(df_train_debug)} rows.")

    # 2. Validation Data
    orig_val_path = "./metadata/val.csv"
    df_val = pd.read_csv(orig_val_path)
    # Take 8 samples for validation
    df_val_debug = df_val.head(8).copy()
    df_val_debug.to_csv(Config.VAL_CSV, index=False)
    print(f"Created {Config.VAL_CSV} with {len(df_val_debug)} rows.")

    # 3. Test Data
    orig_test_path = "./metadata/test.csv"
    df_test = pd.read_csv(orig_test_path)
    # Take 8 samples for testing
    df_test_debug = df_test.head(8).copy()
    df_test_debug.to_csv(Config.TEST_CSV, index=False)
    print(f"Created {Config.TEST_CSV} with {len(df_test_debug)} rows.")


# ==========================================
# Demonstration Tasks
# ==========================================
def demo_dataloader_and_model():
    """
    Demonstrates instantiation of the DataLoader and the Model.
    Verifies tensor shapes and forward pass.
    """
    print("\n--- Demo: DataLoader & Model Architecture ---")

    # 1. Initialize DataLoader (Train mode)
    # This will trigger cache generation for the 16 debug samples
    print("Initializing Train DataLoader...")
    train_loader = get_dataloader(
        mode="train", batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # 2. Fetch one batch
    inputs, targets = next(iter(train_loader))
    x_eeg, x_spec = inputs

    print(f"Batch shapes retrieved:")
    print(f"  EEG Input: {x_eeg.shape} (Expected: [{Config.BATCH_SIZE}, 19, 128, 256])")
    print(
        f"  Spec Input: {x_spec.shape} (Expected: [{Config.BATCH_SIZE}, 4, 256, 256])"
    )
    print(f"  Targets: {targets.shape} (Expected: [{Config.BATCH_SIZE}, 6])")

    # Assertions for Data
    assert x_eeg.shape == (Config.BATCH_SIZE, 19, 128, 256), "EEG tensor shape mismatch"
    assert x_spec.shape == (
        Config.BATCH_SIZE,
        4,
        256,
        256,
    ), "Spectrogram tensor shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE, 6), "Target tensor shape mismatch"

    # 3. Initialize Model
    print("Initializing AttentiveDualScaleNetwork...")
    device = torch.device("cpu")  # Use CPU for shape verification to be safe/simple
    model = AttentiveDualScaleNetwork().to(device)

    # 4. Forward Pass
    print("Performing forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model((x_eeg.to(device), x_spec.to(device)))

    print(f"Output Logits shape: {logits.shape}")

    # Assertions for Model
    assert logits.shape == (Config.BATCH_SIZE, 6), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN logits"

    print("DataLoader and Model verification successful.")


def demo_training_pipeline():
    """
    Runs the full training loop using the debug dataset.
    """
    print("\n--- Demo: Training Pipeline ---")

    # Run training
    # debug=True usually reduces epochs in the library, but we also manually set EPOCHS=1
    # load_cached_data=True allows it to pick up the cache we generated in the previous step
    run_training(debug=True, load_cached_data=True)

    # Verify output
    if os.path.exists(Config.MODEL_PATH):
        file_size = os.path.getsize(Config.MODEL_PATH)
        print(
            f"Training successful. Model saved to {Config.MODEL_PATH} ({file_size / 1024 / 1024:.2f} MB)"
        )
    else:
        raise AssertionError("Training failed: best_model.pth was not created.")


def demo_inference_pipeline():
    """
    Runs the inference pipeline using the trained model and debug test set.
    """
    print("\n--- Demo: Inference Pipeline ---")

    # Run inference
    submission_df = predict(debug=True, load_cached_data=False)

    # Verify output file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Inference failed: submission.csv was not created.")

    # Verify DataFrame content
    expected_rows = 8  # We put 8 rows in test_debug.csv
    if len(submission_df) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"
        )

    # Verify columns
    expected_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    if list(submission_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch.\nExpected: {expected_cols}\nGot: {list(submission_df.columns)}"
        )

    # Verify probabilities sum to 1 (approx)
    vote_cols = expected_cols[1:]
    sums = submission_df[vote_cols].sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise AssertionError("Predicted probabilities do not sum to 1.0")

    print("Inference verification successful.")
    print("Sample Submission:")
    print(submission_df.head())


if __name__ == "__main__":
    try:
        # 1. Prepare Environment
        setup_environment()
        create_debug_data()

        # 2. Run Demos
        demo_dataloader_and_model()
        demo_training_pipeline()
        demo_inference_pipeline()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
