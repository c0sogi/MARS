import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil
import warnings

# Import provided library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.trainer as trainer_lib
import library.inference as inference_lib


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print(">>> Setting up demo environment...")

    # Define paths for mini datasets
    working_dir = "./working/demo_run"
    os.makedirs(working_dir, exist_ok=True)

    mini_train_meta = os.path.join(working_dir, "mini_train.csv")
    mini_val_meta = os.path.join(working_dir, "mini_val.csv")
    mini_test_meta = os.path.join(working_dir, "mini_test.csv")

    # Create mini metadata files (subset of original) to speed up processing
    # We take top 4 samples to ensure batch_size=2 works (if we set it) or default batch_size=8 handles it (padding)
    try:
        df_train = pd.read_csv(config.TRAIN_METADATA_PATH).head(4)
        df_val = pd.read_csv(config.VAL_METADATA_PATH).head(4)
        df_test = pd.read_csv(config.TEST_METADATA_PATH).head(4)

        df_train.to_csv(mini_train_meta, index=False)
        df_val.to_csv(mini_val_meta, index=False)
        df_test.to_csv(mini_test_meta, index=False)
        print(f"Created mini metadata files in {working_dir}")
    except FileNotFoundError:
        print("Error: Original metadata files not found. Ensure ./metadata exists.")
        sys.exit(1)

    # Monkey-patch config and library modules to use these mini files and reduce load
    # This allows us to use the library functions as-is but with our demo configuration

    # Paths
    config.TRAIN_METADATA_PATH = mini_train_meta
    config.VAL_METADATA_PATH = mini_val_meta
    config.TEST_METADATA_PATH = mini_test_meta

    config.TRAIN_CACHE_PATH = os.path.join(working_dir, "cache", "train_data.npz")
    config.VAL_CACHE_PATH = os.path.join(working_dir, "cache", "val_data.npz")
    config.TEST_CACHE_PATH = os.path.join(working_dir, "cache", "test_data.npz")

    config.BEST_MODEL_PATH = os.path.join(working_dir, "checkpoints", "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(
        working_dir, "submission", "demo_submission.csv"
    )

    # Ensure directories exist
    os.makedirs(os.path.dirname(config.TRAIN_CACHE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.BEST_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    # Hyperparameters for speed
    config.BATCH_SIZE = 2
    trainer_lib.NUM_EPOCHS = 2  # Override in trainer module
    trainer_lib.BATCH_SIZE = (
        2  # Ensure trainer uses consistent batch size if referenced
    )
    trainer_lib.BEST_MODEL_PATH = config.BEST_MODEL_PATH  # Sync path
    inference_lib.BEST_MODEL_PATH = config.BEST_MODEL_PATH
    inference_lib.SUBMISSION_PATH = config.SUBMISSION_PATH

    # Set seed for reproducibility
    utils.set_seed(42)
    print("Configuration patched for speed and demo paths.")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Utility Functions...")

    # Test Levenshtein Score
    # Case 1: Perfect match
    score_perfect = utils.levenshtein_score([[1, 2, 3]], [[1, 2, 3]])
    assert score_perfect == 0.0, f"Expected score 0.0, got {score_perfect}"

    # Case 2: One insertion (Target len 2, distance 1) -> Score 0.5
    score_mismatch = utils.levenshtein_score([[1, 2, 3]], [[1, 2]])
    # Distance is 1 (delete 3), target length is 2. Score = 1/2 = 0.5
    assert score_mismatch == 0.5, f"Expected score 0.5, got {score_mismatch}"

    print("Levenshtein score logic verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loading (GestureDataset)...")

    # This will trigger processing from scratch since cache doesn't exist yet
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()

    # Check batch structure
    batch = next(iter(train_loader))
    xs, ys, lengths, ids = batch

    # Validate shapes
    # xs: (Batch, Time, Features=133)
    assert xs.dim() == 3, "Input tensor must be 3D (Batch, Time, Feat)"
    assert (
        xs.shape[2] == config.INPUT_SIZE
    ), f"Feature dim should be {config.INPUT_SIZE}, got {xs.shape[2]}"
    assert (
        xs.shape[0] == config.BATCH_SIZE
    ), f"Batch size should be {config.BATCH_SIZE}, got {xs.shape[0]}"

    # ys: (Batch, Time)
    assert ys.dim() == 2, "Target tensor must be 2D (Batch, Time)"
    assert ys.shape[0] == config.BATCH_SIZE

    # lengths: (Batch)
    assert lengths.shape[0] == config.BATCH_SIZE

    print(f"Data batch verified. Input shape: {xs.shape}, Target shape: {ys.shape}")

    # Verify caching mechanism
    # Re-instantiating dataset should be fast and load from the .npz created above
    print("Verifying cache loading...")
    ds_cached = data_loader.GestureDataset(
        config.TRAIN_METADATA_PATH,
        is_train=True,
        load_cached=True,
        cache_path=config.TRAIN_CACHE_PATH,
    )
    assert len(ds_cached) == 4, "Cached dataset size mismatch"
    print("Cache loading verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.HybridGestureNet().to(device)

    # Forward pass with the batch we loaded
    xs = xs.to(device)
    lengths = lengths.to(
        device
    )  # Lengths usually stay on CPU for pack_padded but model handles it

    logits1, logits2 = model(xs, lengths)

    # Check output shapes
    # (Batch, Time, NumClasses)
    assert logits1.shape == (config.BATCH_SIZE, xs.shape[1], config.NUM_CLASSES)
    assert logits2.shape == (config.BATCH_SIZE, xs.shape[1], config.NUM_CLASSES)

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Training Loop...")

    trainer = trainer_lib.Trainer(model, train_loader, val_loader, device)

    # Run fit (patched to 2 epochs)
    trainer.fit()

    # Check if model checkpoint was saved
    assert os.path.exists(
        config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print(f"Training completed. Checkpoint saved at {config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 6. Verify Inference
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Inference pipeline...")

    # Run inference
    inference_lib.generate_submission(device=str(device))

    # Check submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission content format
    with open(config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    assert (
        len(lines) == 4
    ), f"Expected 4 lines in submission (4 test samples), got {len(lines)}"

    # Check format of first line: SessionID,Labels...
    first_line = lines[0].strip()
    parts = first_line.split(",")
    assert (
        len(parts) >= 1
    ), "Submission line format incorrect (must have at least SessionID)"
    # SessionID should match the pattern in mini_test.csv (e.g., Sample00300)
    # We can check if it starts with 'Sample'
    assert parts[0].startswith("Sample"), f"SessionID format unexpected: {parts[0]}"

    print(f"Inference verified. Output:\n{lines[0].strip()} ...")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    try:
        run_demo()
    except AssertionError as e:
        print(f"\n!!! Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
