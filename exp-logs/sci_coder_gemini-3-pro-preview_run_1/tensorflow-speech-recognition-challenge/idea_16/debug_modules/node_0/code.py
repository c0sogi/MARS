import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from unittest.mock import patch

# Import library modules
from library.config import Config, set_seed
import library.trainer as trainer_module
from library.dataset import SpeechCommandDataset, MixupCollate
from library.model import DilatedEfficientNet
from library.utils import map_fine_grained_to_12_class
from library.trainer import Trainer


def run_demo():
    print("=== Starting Speech Command Recognition Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_PATH = "./working/demo_submission/demo_submission.csv"

    # Create working directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")
    # Check mapping logic
    assert map_fine_grained_to_12_class("yes") == "yes", "Failed: Target label mapping"
    assert (
        map_fine_grained_to_12_class("bed") == "unknown"
    ), "Failed: Auxiliary label mapping"
    assert (
        map_fine_grained_to_12_class("silence") == "silence"
    ), "Failed: Silence label mapping"
    print("    Label mapping logic verified.")

    # -------------------------------------------------------------------------
    # 3. Data Preparation (Subsetting)
    # -------------------------------------------------------------------------
    print("\n[3] Preparing Data Subsets...")

    # Load real metadata to sample from
    # We use the library function directly here to get the full DF first
    full_df = trainer_module.load_or_create_metadata(load_cached_data=False)

    # Create a tiny training/validation subset (e.g., 20 samples total)
    # Ensure we have at least a few classes
    subset_train = full_df[full_df["split"] == "train"].sample(
        n=32, random_state=Config.SEED
    )
    subset_val = full_df[full_df["split"] == "val"].sample(
        n=16, random_state=Config.SEED
    )
    subset_df = pd.concat([subset_train, subset_val]).reset_index(drop=True)

    print(f"    Created subset with {len(subset_df)} samples.")

    # Create a tiny test subset
    real_test_df = pd.read_csv(Config.TEST_METADATA)
    subset_test_df = real_test_df.sample(n=10, random_state=Config.SEED)

    # Save temporary test metadata to overwrite Config path temporarily
    temp_test_meta_path = "./working/test_small.csv"
    subset_test_df.to_csv(temp_test_meta_path, index=False)
    Config.TEST_METADATA = temp_test_meta_path
    print(f"    Created test subset with {len(subset_test_df)} samples.")

    # -------------------------------------------------------------------------
    # 4. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Dataset and DataLoader...")

    # Fit a label encoder on the subset for testing
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(subset_df["fine_label"])

    # Instantiate Dataset
    ds = SpeechCommandDataset(subset_df, label_encoder=le, is_train=True)

    # Check __getitem__
    spec, label = ds[0]
    print(f"    Sample Spectrogram Shape: {spec.shape}")
    print(f"    Sample Label: {label}")

    # Expected shape: [1, N_MELS, TIME_STEPS]
    # TIME_STEPS = sr * duration / hop_length = 16000 * 1.0 / 160 = 100
    # N_MELS = 128
    expected_freq = Config.N_MELS
    expected_time = int(Config.SAMPLE_RATE * Config.DURATION / Config.HOP_LENGTH)

    # Allow small margin in time dimension due to padding/cropping logic rounding
    assert spec.shape[0] == 1, "Spectrogram should have 1 channel"
    assert spec.shape[1] == expected_freq, f"Expected {expected_freq} mels"
    assert (
        abs(spec.shape[2] - expected_time) <= 1
    ), f"Expected approx {expected_time} time steps"

    # Check Mixup Collate
    collate_fn = MixupCollate(alpha=1.0)
    batch = [ds[i] for i in range(4)]
    mixed_imgs, target_a, target_b, lam = collate_fn(batch)

    assert mixed_imgs.shape[0] == 4
    assert mixed_imgs.shape[1:] == spec.shape
    print("    Dataset and MixupCollate verified.")

    # -------------------------------------------------------------------------
    # 5. Model Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Model Architecture...")
    num_classes = len(le.classes_)
    model = DilatedEfficientNet(num_classes=num_classes)
    model.eval()

    # Forward pass with the mixed batch
    with torch.no_grad():
        outputs = model(mixed_imgs)

    print(f"    Model Output Shape: {outputs.shape}")
    assert outputs.shape == (4, num_classes), "Model output shape mismatch"
    print("    Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Trainer (Train & Validation)...")

    # We patch 'load_or_create_metadata' in library.trainer to return our subset
    # This ensures the Trainer uses our small dataset without modifying library code
    with patch("library.trainer.load_or_create_metadata", return_value=subset_df):
        trainer = Trainer()
        trainer.train()

    # Check if best model was saved
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Best model file was not saved."
    print("    Training completed and model saved.")

    # -------------------------------------------------------------------------
    # 7. Inference & Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n[7] Generating Submission...")

    # We reuse the trainer instance.
    # Note: Trainer.generate_submission reads Config.TEST_METADATA which we overrode earlier.
    # It also needs to refit the encoder or use the existing one.
    # The trainer instance already has a fitted encoder from .train().

    trainer.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Rows: {len(sub_df)}")
    print(f"    Submission Columns: {sub_df.columns.tolist()}")

    assert len(sub_df) == len(subset_test_df), "Submission row count mismatch"
    assert (
        "fname" in sub_df.columns and "label" in sub_df.columns
    ), "Submission columns mismatch"

    # Check if labels are within the allowed 12 classes
    valid_labels = Config.TARGET_LABELS.union(
        {Config.SILENCE_LABEL, Config.UNKNOWN_LABEL}
    )
    invalid_preds = sub_df[~sub_df["label"].isin(valid_labels)]
    if not invalid_preds.empty:
        raise AssertionError(
            f"Found invalid labels in submission: {invalid_preds['label'].unique()}"
        )

    print("    Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
