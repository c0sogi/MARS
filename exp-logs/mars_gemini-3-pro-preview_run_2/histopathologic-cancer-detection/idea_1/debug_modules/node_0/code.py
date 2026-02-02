import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.dataset import PathologyDataset, get_transforms
from library.model import TumorClassifier
from library.train import run_training
from library.predict import generate_submission
from library.utils import set_seed, compute_metrics


def main():
    # --- 1. Setup & Configuration for Demonstration ---
    print("--- Setting up demonstration configuration ---")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Modify Config for speed (runtime overrides)
    # Note: Function default args (like in run_training) are bound at import time,
    # so we must pass explicit args to functions, but we modify Config for
    # classes that access it dynamically (like Dataset).
    Config.DEBUG_SAMPLE_SIZE = 50  # Reduce debug size for ultra-fast execution
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("Configuration set. Debug sample size:", Config.DEBUG_SAMPLE_SIZE)

    # --- 2. Dataset & Transforms Verification ---
    print("\n--- Verifying Dataset and Transforms ---")

    # Initialize datasets in debug mode
    train_ds = PathologyDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        debug=True,
    )
    val_ds = PathologyDataset(
        metadata_path=Config.VAL_METADATA_PATH,
        transform=get_transforms("val"),
        debug=True,
    )

    # Verify length
    # We expect length to be min(total_rows, Config.DEBUG_SAMPLE_SIZE)
    # Since we set DEBUG_SAMPLE_SIZE to 50, and metadata has thousands, it should be 50.
    assert (
        len(train_ds) == Config.DEBUG_SAMPLE_SIZE
    ), f"Train dataset length {len(train_ds)} does not match debug size {Config.DEBUG_SAMPLE_SIZE}"

    # Verify item structure
    sample_img, sample_label = train_ds[0]

    # Check types
    assert isinstance(sample_img, torch.Tensor), "Dataset image is not a torch.Tensor"
    assert isinstance(sample_label, torch.Tensor), "Dataset label is not a torch.Tensor"

    # Check Image Shape: (C, H, W) -> (3, 64, 64) based on Config.CROP_SIZE=64
    expected_shape = (3, Config.CROP_SIZE, Config.CROP_SIZE)
    assert (
        sample_img.shape == expected_shape
    ), f"Image shape {sample_img.shape} mismatch. Expected {expected_shape}"

    # Check Label Shape/Type: Should be scalar float32 (0.0 or 1.0)
    assert sample_label.ndim == 0, "Label should be a scalar"
    assert sample_label.dtype == torch.float32, "Label should be float32"

    print("Dataset verification passed.")

    # --- 3. Model Verification ---
    print("\n--- Verifying Model Architecture ---")

    # Initialize model (no pretrained weights needed for structure check)
    model = TumorClassifier(pretrained=False)
    model.eval()

    # Create dummy input batch
    dummy_batch_size = 4
    dummy_input = torch.randn(dummy_batch_size, 3, Config.CROP_SIZE, Config.CROP_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check Output Shape: (Batch_Size, Num_Classes) -> (4, 1)
    assert output.shape == (
        dummy_batch_size,
        Config.NUM_CLASSES,
    ), f"Model output shape {output.shape} mismatch. Expected {(dummy_batch_size, Config.NUM_CLASSES)}"

    print("Model verification passed.")

    # --- 4. Training Loop Demonstration ---
    print("\n--- Running Training Demonstration ---")

    # Run training for 1 epoch on debug subset
    # Explicitly pass arguments to override defaults bound at import
    best_model_path = run_training(
        epochs=1,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=True,
    )

    # Verify checkpoint existence
    assert os.path.exists(best_model_path), f"Checkpoint not found at {best_model_path}"
    print(f"Training finished. Checkpoint saved: {best_model_path}")

    # --- 5. Prediction Demonstration ---
    print("\n--- Running Prediction Demonstration ---")

    # Generate submission using the trained model
    generate_submission(
        checkpoint_path=best_model_path,
        output_path=Config.PREDICTION_FILE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=True,
    )

    # Verify submission file
    assert os.path.exists(Config.PREDICTION_FILE), "Submission file not created"

    df_sub = pd.read_csv(Config.PREDICTION_FILE)

    # Check columns
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "label" in df_sub.columns, "Submission missing 'label' column"

    # Check row count (should match debug size)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count {len(df_sub)} mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}"

    # Check probability range
    probs = df_sub["label"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Predictions contain values outside [0, 1]"

    print(f"Prediction verification passed. Output at {Config.PREDICTION_FILE}")

    # --- 6. Metrics Verification ---
    print("\n--- Verifying Metrics Utility ---")

    # Test compute_metrics with known values
    y_true_test = np.array([0, 0, 1, 1])
    y_pred_test = np.array([0.1, 0.4, 0.35, 0.8])
    # AUC Calculation:
    # Pairs: (0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8)
    # Positives: 0.35, 0.8
    # Negatives: 0.1, 0.4
    # Comparisons:
    # 0.35 > 0.1 (Win)
    # 0.35 < 0.4 (Loss)
    # 0.8 > 0.1 (Win)
    # 0.8 > 0.4 (Win)
    # Total Wins = 3, Total Comparisons = 2*2 = 4. AUC = 0.75

    auc_score = compute_metrics(y_true_test, y_pred_test)
    assert (
        abs(auc_score - 0.75) < 1e-6
    ), f"AUC calculation incorrect. Got {auc_score}, expected 0.75"

    print("Metrics verification passed.")

    print("\n=== All System Checks Completed Successfully ===")


if __name__ == "__main__":
    main()
