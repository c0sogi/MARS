import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed
import library.data  # Import module to patch tqdm
from library.data import get_dataloaders
from library.model import SiameseEfficientNet
from library.train import run_training
from library.predict import predict_submission


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("Initializing demonstration...")

    # Suppress warnings
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Monkey-patch tqdm to disable progress bars in library.data
    def silent_tqdm(iterable, *args, **kwargs):
        return iterable

    library.data.tqdm = silent_tqdm

    # Override Config for a fast demonstration
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to use the demo directory
    Config.CACHE_TRAIN_DATA = os.path.join(Config.WORKING_DIR, "demo_train_data.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(
        Config.WORKING_DIR, "demo_train_labels.npy"
    )
    Config.CACHE_VAL_DATA = os.path.join(Config.WORKING_DIR, "demo_val_data.npy")
    Config.CACHE_VAL_LABELS = os.path.join(Config.WORKING_DIR, "demo_val_labels.npy")
    Config.CACHE_TEST_DATA = os.path.join(Config.WORKING_DIR, "demo_test_data.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "demo_test_ids.npy")

    # Update submission paths
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Model checkpoint path (hardcoded in train.py to use Config.WORKING_DIR)
    # We rely on Config.WORKING_DIR being updated above.

    # Hyperparameters for speed
    Config.DEBUG_SAMPLE_SIZE = 5  # Only process 5 samples per split
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch training
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure reproducibility
    set_seed(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("Verifying data pipeline...")

    # Load data (this triggers processing and caching)
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(load_cached=False)

    # Check Train Loader
    try:
        texture_batch, context_batch, label_batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # Verify Shapes
    # Expected: (Batch, 12, 224, 224)
    expected_shape = (
        Config.BATCH_SIZE,
        Config.NUM_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )

    assert (
        texture_batch.shape == expected_shape
    ), f"Texture batch shape mismatch. Expected {expected_shape}, got {texture_batch.shape}"
    assert (
        context_batch.shape == expected_shape
    ), f"Context batch shape mismatch. Expected {expected_shape}, got {context_batch.shape}"
    assert label_batch.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Label batch shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {label_batch.shape}"

    print("Data pipeline verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Logic Verification
    # --------------------------------------------------------------------------
    print("Verifying model architecture...")

    device = torch.device("cpu")  # Use CPU for quick logic check
    model = SiameseEfficientNet().to(device)
    model.eval()

    with torch.no_grad():
        # Run a dummy forward pass
        output = model(texture_batch.to(device), context_batch.to(device))

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    print("Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Execution
    # --------------------------------------------------------------------------
    print("Executing training loop (1 epoch)...")

    # Run training using the cached data we just generated
    best_auc = run_training(load_cached=True)

    # Verify model artifact creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Training failed to generate model file at {best_model_path}"
        )

    print(f"Training completed. Best AUC: {best_auc}")

    # --------------------------------------------------------------------------
    # 5. Inference & Submission
    # --------------------------------------------------------------------------
    print("Executing inference and generating submission...")

    predict_submission(load_cached=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify submission content
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission file missing required columns."

    # Since we used DEBUG_SAMPLE_SIZE=5 for test set generation, we expect roughly 5 rows.
    # Note: process_and_cache_data uses head(DEBUG_SAMPLE_SIZE).
    # However, if some files were corrupt/missing in the first 5, it might be less.
    # Given the dataset description, corruption is unlikely in the first few,
    # but we just check it's not empty.
    assert len(df_sub) > 0, "Submission file is empty."

    # Verify IDs match the test_ids returned by loader
    # (test_ids might be a subset if DEBUG_SAMPLE_SIZE was applied)
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count ({len(df_sub)}) does not match test ID count ({len(test_ids)})."

    print("Submission generated and verified successfully.")
    print(f"Demo completed. Outputs stored in {Config.WORKING_DIR}")


if __name__ == "__main__":
    main()
