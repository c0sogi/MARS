import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data import process_audio_file, get_spectrogram_transform
from library.model import WhaleClassifier
from library.losses import get_loss_module, MixupLoss
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Whale Detection Demo Script ===")

    # 1. Setup
    seed_everything(Config.SEED)

    # Define paths for demo subset data
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_csv = os.path.join(demo_dir, "train_subset.csv")
    demo_val_csv = os.path.join(demo_dir, "val_subset.csv")
    demo_test_csv = os.path.join(demo_dir, "test_subset.csv")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_submission_dir = os.path.join(demo_dir, "submission")
    demo_submission_path = os.path.join(demo_submission_dir, "submission.csv")
    demo_model_path = os.path.join(demo_dir, "best_model.pth")

    # 2. Create Data Subsets
    print("\n[Step 1] Creating data subsets for rapid execution...")

    # Load original metadata
    full_train = pd.read_csv(Config.TRAIN_CSV)
    full_val = pd.read_csv(Config.VAL_CSV)
    full_test = pd.read_csv(Config.TEST_CSV)

    # Sample subsets (ensure we have both classes in train if possible)
    # Taking 50 train, 20 val, 20 test
    subset_train = full_train.sample(n=50, random_state=Config.SEED).reset_index(
        drop=True
    )
    subset_val = full_val.sample(n=20, random_state=Config.SEED).reset_index(drop=True)
    subset_test = full_test.sample(n=20, random_state=Config.SEED).reset_index(
        drop=True
    )

    # Save subsets
    subset_train.to_csv(demo_train_csv, index=False)
    subset_val.to_csv(demo_val_csv, index=False)
    subset_test.to_csv(demo_test_csv, index=False)

    print(
        f"Created subsets: Train={len(subset_train)}, Val={len(subset_val)}, Test={len(subset_test)}"
    )

    # 3. Override Configuration
    print("\n[Step 2] Overriding Config for demo...")

    # Modify Config attributes globally
    Config.TRAIN_CSV = demo_train_csv
    Config.VAL_CSV = demo_val_csv
    Config.TEST_CSV = demo_test_csv
    Config.CACHE_DIR = demo_cache_dir
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = demo_submission_path
    Config.BEST_MODEL_PATH = demo_model_path

    # Reduce compute requirements
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2  # Reduce worker overhead for small data

    # Ensure directories exist (Config.setup() usually does this, but we changed paths)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 4. Component Validation
    print("\n[Step 3] Validating individual components...")

    # A. Test Data Processing
    print("  -> Testing audio processing...")
    mel_transform = get_spectrogram_transform()
    sample_file = subset_train.iloc[0]["file_path"]
    spectrogram = process_audio_file(sample_file, mel_transform)

    assert isinstance(spectrogram, np.ndarray), "Processed audio must be a numpy array"
    assert spectrogram.shape == (
        1,
        224,
        224,
    ), f"Expected shape (1, 224, 224), got {spectrogram.shape}"
    print("     Audio processing check passed.")

    # B. Test Model Architecture
    print("  -> Testing model architecture...")
    model = WhaleClassifier()
    dummy_input = torch.randn(4, 1, 224, 224)  # Batch of 4
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        4,
        1,
    ), f"Expected model output shape (4, 1), got {output.shape}"
    print("     Model architecture check passed.")

    # C. Test Loss Function
    print("  -> Testing loss functions...")
    # Test WeightedBCELoss
    criterion = get_loss_module()
    dummy_targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = criterion(output, dummy_targets)
    assert loss.dim() == 0, "Loss should be a scalar"

    # Test MixupLoss
    mixup_criterion = MixupLoss(criterion)
    # y_a and y_b for mixup
    y_a = dummy_targets
    y_b = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    lam = 0.7
    mixup_loss = mixup_criterion(output, y_a, y_b, lam)
    assert mixup_loss.dim() == 0, "Mixup loss should be a scalar"
    print("     Loss function check passed.")

    # 5. Full Pipeline Execution
    print("\n[Step 4] Running full training pipeline on subset...")

    # We set load_cached_data=False to force processing of our new subset
    run_training(epochs=Config.EPOCHS, load_cached_data=False, patience=2)

    # 6. Result Verification
    print("\n[Step 5] Verifying artifacts...")

    # Check Model
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"  -> Found best model at {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Best model checkpoint was not created.")

    # Check Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"  -> Found submission file at {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  -> Submission shape: {df_sub.shape}")

        expected_rows = len(subset_test)
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"
        assert (
            "clip" in df_sub.columns and "probability" in df_sub.columns
        ), "Submission missing required columns"
        assert (
            df_sub["probability"].dtype == float
            or df_sub["probability"].dtype == np.float32
            or df_sub["probability"].dtype == np.float64
        ), "Probability column should be float"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
