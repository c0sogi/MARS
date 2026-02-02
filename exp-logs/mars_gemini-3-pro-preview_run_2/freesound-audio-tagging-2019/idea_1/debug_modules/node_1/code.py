import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.dataset import get_dataloaders
from library.model import AudioMobileNet
from library.utils import calculate_lwlrap, set_seed
from library.engine import run


def create_mini_test_set():
    """Creates a smaller test csv to speed up the inference step."""
    full_test_df = pd.read_csv(Config.TEST_CSV)
    mini_test_df = full_test_df.head(20)  # Use only 20 samples for testing

    mini_test_path = os.path.join(Config.WORKING_DIR, "test_mini.csv")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    mini_test_df.to_csv(mini_test_path, index=False)

    return mini_test_path


def main():
    print("==== Starting Audio Tagging Demonstration ====")

    # 1. Configure for Speed and Debugging
    print("\n[1] Configuring environment...")
    Config.DEBUG = True  # Limits train/val samples (100/50)
    Config.MAX_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this quick test

    # Override Test CSV to use a mini subset for fast inference
    Config.TEST_CSV = create_mini_test_set()
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Test CSV redirected to: {Config.TEST_CSV}")

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # 2. Verify Data Loading
    print("\n[2] Verifying Data Loading...")
    # load_cached_data=False ensures we use the new Config.TEST_CSV and DEBUG settings
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch
    images, targets, fnames = next(iter(train_loader))

    print(f"    Batch Image Shape: {images.shape}")
    print(f"    Batch Target Shape: {targets.shape}")

    # Assertions
    assert images.dim() == 4, "Images must be 4D: (B, C, F, T)"
    assert (
        images.shape[1] == Config.IN_CHANNELS
    ), f"Expected {Config.IN_CHANNELS} channel(s)"
    assert (
        targets.shape[1] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes"
    assert len(fnames) == images.shape[0], "Mismatch between batch size and filenames"
    print("    Data Loading verification passed.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    model = AudioMobileNet().to(device)

    # Create dummy input: (Batch=2, Channel=1, Freq=128, Time=313)
    # Time=313 corresponds roughly to 10s at the given hop length, but model handles variable length
    dummy_input = torch.randn(2, 1, 128, 313).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"
    print("    Model verification passed.")

    # 4. Verify Metric (LWLRAP)
    print("\n[4] Verifying Metric (LWLRAP)...")
    # Synthetic ground truth (3 classes)
    y_true_dummy = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
    # Synthetic predictions
    y_score_dummy = np.array([[0.9, 0.1, 0.0], [0.2, 0.8, 0.6], [0.1, 0.1, 0.9]])

    score = calculate_lwlrap(y_true_dummy, y_score_dummy)
    print(f"    Calculated LWLRAP: {score:.4f}")
    assert 0.0 <= score <= 1.0, "LWLRAP score must be between 0 and 1"
    print("    Metric verification passed.")

    # 5. Execute Training Engine
    print("\n[5] Running Training Engine (Train -> Val -> Predict)...")
    # This runs the training loop, validation, and generates submission
    run(
        model,
        train_loader,
        val_loader,
        test_loader,
        epochs=Config.MAX_EPOCHS,
        device=device,
    )

    # 6. Verify Submission Output
    print("\n[6] Verifying Submission...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Shape: {sub_df.shape}")

    # Check rows: Should match the mini test set size (20)
    expected_rows = 20
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Check columns: fname + 80 classes
    expected_cols = Config.NUM_CLASSES + 1
    assert (
        sub_df.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, got {sub_df.shape[1]}"

    # Check first column is fname
    assert sub_df.columns[0] == "fname", "First column must be 'fname'"

    print("    Submission verification passed.")
    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
