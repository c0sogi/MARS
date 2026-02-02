import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
import library.config
import library.utils
import library.dataset
import library.model
import library.train


def main():
    print("=== Starting ISIC Task Demonstration & Verification ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Patching
    # -------------------------------------------------------------------------
    # We patch the configuration to run a fast demo (DEBUG mode, 1 Epoch)
    # Since modules import constants using 'from x import y', we must patch
    # the variables in the destination modules as well.
    print("[1] Patching configuration for fast execution...")

    # Enable Debug Mode (uses 1000 samples)
    library.config.DEBUG = True
    library.dataset.DEBUG = True

    # Reduce Epochs
    library.config.EPOCHS = 1
    library.train.EPOCHS = 1

    # Reduce Batch Size for safety/speed in demo
    demo_batch_size = 16
    library.config.BATCH_SIZE = demo_batch_size
    library.dataset.BATCH_SIZE = demo_batch_size
    library.train.BATCH_SIZE = demo_batch_size

    # Ensure working directory exists
    os.makedirs(library.config.WORKING_DIR, exist_ok=True)

    print("    DEBUG mode enabled.")
    print("    EPOCHS set to 1.")
    print("    BATCH_SIZE set to 16.")
    print("    Configuration patched successfully.\n")

    # -------------------------------------------------------------------------
    # 2. Utility Verification
    # -------------------------------------------------------------------------
    print("[2] Verifying Utility Functions...")

    # Test Seed
    library.utils.seed_everything(42)

    # Test ROC AUC
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    score = library.utils.calculate_roc_auc(y_true, y_pred)

    # Expected: 0.75
    # Explanation: Pairs are (0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8)
    # Positives: 0.35, 0.8. Negatives: 0.1, 0.4.
    # 0.35 > 0.1 (Correct), 0.35 < 0.4 (Incorrect)
    # 0.8 > 0.1 (Correct), 0.8 > 0.4 (Correct)
    # Total 3/4 correct pairs = 0.75
    assert np.isclose(
        score, 0.75
    ), f"ROC AUC calculation failed. Expected 0.75, got {score}"
    print(f"    ROC AUC check passed: {score}")
    print("    Utils verified.\n")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("[3] Verifying Data Pipeline...")

    # We force load_cached_data=False to ensure the pipeline runs from scratch for the demo
    train_loader, val_loader, test_loader = library.dataset.get_dataloaders(
        load_cached_data=False,
        batch_size=demo_batch_size,
        num_workers=2,  # Reduced workers for demo script
    )

    # Verify Dataset Size (DEBUG mode should limit to ~1000 samples)
    # train_df is sampled from 1000 rows, split into train/val implicitly by metadata?
    # Actually, get_dataloaders loads train.csv and heads(1000).
    # train.csv has 23k rows. Head(1000) -> 1000 rows.
    dataset_len = len(train_loader.dataset)
    print(f"    Train Dataset Length: {dataset_len}")
    assert dataset_len <= 1000, "DEBUG mode did not reduce dataset size correctly."

    # Fetch one batch
    images, tabular, targets = next(iter(train_loader))

    # Verify Shapes
    # Image: [B, 3, 256, 256]
    assert images.shape == (
        demo_batch_size,
        3,
        256,
        256,
    ), f"Image shape mismatch. Expected {(demo_batch_size, 3, 256, 256)}, got {images.shape}"

    # Tabular: [B, N_feats]. We need to check N_feats.
    # Based on pipeline: age(1) + sex(3 approx) + site(7 approx) ~ 10-15 features
    n_tab_feats = tabular.shape[1]
    assert tabular.shape[0] == demo_batch_size, "Tabular batch size mismatch."

    # Targets: [B]
    assert targets.shape == (demo_batch_size,), "Target shape mismatch."

    print(
        f"    Batch Shapes Verified: Images {images.shape}, Tabular {tabular.shape}, Targets {targets.shape}"
    )
    print("    Data Pipeline verified.\n")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("[4] Verifying Model Architecture...")

    device = library.config.DEVICE
    model = library.model.SwinTransformerGLU(
        tabular_input_dim=n_tab_feats, pretrained=False
    )
    model.to(device)

    # Move batch to device
    images = images.to(device)
    tabular = tabular.to(device)

    # Forward Pass
    logits = model(images, tabular)

    # Verify Output Shape: [B, 1] (Binary classification logits)
    # Note: The model definition uses NUM_CLASSES=1 in config.
    # library.model.SwinTransformerGLU.head is Linear(dim, NUM_CLASSES)
    # Output should be [B, 1]
    assert logits.shape == (
        demo_batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(demo_batch_size, 1)}, got {logits.shape}"

    print(f"    Forward pass successful. Output shape: {logits.shape}")
    print("    Model verified.\n")

    # -------------------------------------------------------------------------
    # 5. Full Training Loop Verification
    # -------------------------------------------------------------------------
    print("[5] Running Full Training Loop (Integration Test)...")
    print("    This will run 1 epoch on the debug subset.")

    # We call run_training. We've already patched EPOCHS and DEBUG.
    # We set load_cached_data=True to use the features we just generated in step 3
    # (get_dataloaders in step 3 saved them to disk).

    try:
        library.train.run_training(load_cached_data=True, patience=1)
    except Exception as e:
        print(f"    Training failed with error: {e}")
        raise e

    print("    Training loop completed successfully.\n")

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("[6] Verifying Submission File...")

    submission_path = library.config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission file loaded. Rows: {len(sub_df)}")

    # Check columns
    assert (
        "image_name" in sub_df.columns and "target" in sub_df.columns
    ), "Submission file missing required columns."

    # Check length (DEBUG mode affects test set too, so it should be 1000)
    # dataset.py: if DEBUG: test_df = test_df.head(DEBUG_SAMPLE_SIZE)
    expected_len = library.config.DEBUG_SAMPLE_SIZE
    assert (
        len(sub_df) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(sub_df)}"

    # Check values are probabilities (0-1)
    preds = sub_df["target"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions are not valid probabilities (must be between 0 and 1)."

    print("    Submission format verified.")
    print("\n=== All Demonstrations & Verifications Passed ===")


if __name__ == "__main__":
    main()
