import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_mcc, optimize_threshold, FocalLoss
from library.data_processing import get_datasets, get_test_dataset
from library.model import SEARVN
from library.train import train_one_epoch, validate
from library.inference import generate_predictions


def main():
    print("Starting SEA-RVN Library Demo...")

    # =========================================================================
    # 1. Configuration Setup & Overrides for Demo Speed
    # =========================================================================
    print("\n[1] Configuring environment...")

    # Override Config for a fast, lightweight run
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 500  # Process only 500 samples for training
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.THRESHOLD_SEARCH_STEPS = 10  # Reduced precision for speed

    # Setup directories
    Config.setup()

    # Set seed
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # =========================================================================
    # 2. Data Processing Demonstration
    # =========================================================================
    print("\n[2] Testing Data Processing (get_datasets)...")

    # We force load_cached_data=False to ensure the processing logic runs
    # The processing will use the reduced DEBUG_SAMPLES size
    train_ds, val_ds = get_datasets(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size: {len(val_ds)}")

    # Verification
    assert len(train_ds) > 0, "Training dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "scaler_cont.joblib")
    ), "Scaler artifact missing."

    # Inspect a single sample
    x_kin_cont, x_kin_cat, x_vis, y = train_ds[0]
    print(
        f"Sample shapes - KinCont: {x_kin_cont.shape}, KinCat: {x_kin_cat.shape}, Vis: {x_vis.shape}, Target: {y.shape}"
    )

    # =========================================================================
    # 3. Model Initialization & Forward Pass
    # =========================================================================
    print("\n[3] Testing Model Initialization & Forward Pass...")

    device = torch.device(Config.DEVICE)
    model = SEARVN().to(device)

    # Create a dataloader for a single batch
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch = next(iter(train_loader))
    x_kin_cont, x_kin_cat, x_vis, targets = [b.to(device) for b in batch]

    # Forward pass
    logits = model(x_kin_cont, x_kin_cat, x_vis)

    # Verification
    # Output should be (Batch_Size, 1) or (Batch_Size,) depending on squeeze
    print(f"Logits Shape: {logits.shape}")
    assert logits.shape[0] == x_kin_cont.shape[0], "Output batch size mismatch."

    # =========================================================================
    # 4. Training Loop Demonstration
    # =========================================================================
    print("\n[4] Testing Training Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss (1 epoch): {train_loss:.4f}")

    # Validate
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    val_loss, val_mcc, val_thresh = validate(model, val_loader, criterion, device)
    print(
        f"Val Loss: {val_loss:.4f}, MCC: {val_mcc:.4f}, Best Thresh: {val_thresh:.4f}"
    )

    # Verification
    assert not np.isnan(train_loss), "Training loss is NaN."
    assert 0.0 <= val_thresh <= 1.0, "Invalid threshold value."

    # Save model and threshold manually for the inference step
    torch.save(model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth"))
    np.save(os.path.join(Config.WORKING_DIR, "best_threshold.npy"), val_thresh)
    print("Saved best_model.pth and best_threshold.npy for inference demo.")

    # =========================================================================
    # 5. Inference Demonstration
    # =========================================================================
    print("\n[5] Testing Inference Pipeline...")

    # To keep inference fast, we create a small subset of the test metadata
    # and point Config to it.
    full_test_meta = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))
    subset_test_meta_path = os.path.join(Config.WORKING_DIR, "test_subset.csv")
    full_test_meta.head(100).to_csv(subset_test_meta_path, index=False)

    # Override Config to use this subset
    Config.TEST_META = subset_test_meta_path

    # Run Inference
    # force load_cached_data=False to ensure it processes our new subset file
    generate_predictions(load_cached_data=False)

    # Verification
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission generated with {len(sub_df)} rows.")
    assert len(sub_df) == 100, f"Expected 100 rows in submission, got {len(sub_df)}"
    assert (
        "contact_id" in sub_df.columns and "contact" in sub_df.columns
    ), "Submission columns missing."

    # =========================================================================
    # 6. Utility Functions Verification
    # =========================================================================
    print("\n[6] Testing Utilities...")

    # Test MCC
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    mcc = compute_mcc(y_true, y_pred)
    print(f"Calculated MCC: {mcc:.4f}")
    assert -1.0 <= mcc <= 1.0, "MCC out of range."

    # Test Threshold Optimization
    probs = np.array([0.1, 0.4, 0.6, 0.8, 0.3])
    targets = np.array([0, 0, 1, 1, 0])
    best_t, best_s = optimize_threshold(targets, probs, steps=10)
    print(f"Optimized Threshold: {best_t:.2f}, Score: {best_s:.4f}")
    assert best_t > 0, "Threshold optimization failed."

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
