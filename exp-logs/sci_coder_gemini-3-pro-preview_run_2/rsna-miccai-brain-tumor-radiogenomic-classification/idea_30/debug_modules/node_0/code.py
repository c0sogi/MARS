import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------------------------------------------------------
# 1. Setup and Monkey Patching
# -----------------------------------------------------------------------------
# We redirect the cache paths in the library to a demo folder to ensure
# this script is self-contained and doesn't overwrite main experiment files.
DEMO_DIR = "./working/demo_execution"
os.makedirs(DEMO_DIR, exist_ok=True)

import library.data

library.data.CACHE_DIR = DEMO_DIR
library.data.ROI_CACHE_FILE = os.path.join(DEMO_DIR, "roi_cache_demo.parquet")

# Import provided libraries
from library.data import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train import train_one_epoch, validate, set_seed

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Demo Execution...")

    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Prepare Data Subsets (for Speed)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Preparing data subsets...")

    # Define paths
    meta_train_path = "./metadata/train.csv"
    meta_val_path = "./metadata/val.csv"
    meta_test_path = "./metadata/test.csv"

    # Load original metadata
    df_train = pd.read_csv(meta_train_path)
    df_val = pd.read_csv(meta_val_path)
    df_test = pd.read_csv(meta_test_path)

    # Create subsets (take top N rows)
    # Using a small number to ensure the demo finishes in seconds/minutes
    subset_size = 8
    df_train_sub = df_train.head(subset_size).copy()
    df_val_sub = df_val.head(subset_size).copy()
    df_test_sub = df_test.head(subset_size).copy()

    # Save subsets
    subset_meta_dir = os.path.join(DEMO_DIR, "subset_metadata")
    os.makedirs(subset_meta_dir, exist_ok=True)

    sub_train_path = os.path.join(subset_meta_dir, "train.csv")
    sub_val_path = os.path.join(subset_meta_dir, "val.csv")
    sub_test_path = os.path.join(subset_meta_dir, "test.csv")

    df_train_sub.to_csv(sub_train_path, index=False)
    df_val_sub.to_csv(sub_val_path, index=False)
    df_test_sub.to_csv(sub_test_path, index=False)

    print(f"Subsets saved to {subset_meta_dir}")

    # -------------------------------------------------------------------------
    # 3. Instantiate DataLoaders
    # -------------------------------------------------------------------------
    print("\n[Step 2] Initializing DataLoaders...")

    batch_size = 4
    train_loader, val_loader, test_loader = get_dataloaders(
        train_metadata_path=sub_train_path,
        val_metadata_path=sub_val_path,
        test_metadata_path=sub_test_path,
        batch_size=batch_size,
        num_workers=2,
        load_cached_roi=False,  # Force re-computation for the subset
    )

    # Verification: Check batch structure
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {sample_batch.shape}")  # Should be (B, 12, 224, 224)
    print(f"Batch Label Shape: {sample_labels.shape}")  # Should be (B,)

    assert sample_batch.shape == (
        batch_size,
        12,
        224,
        224,
    ), "Incorrect input tensor shape"
    assert sample_labels.shape == (batch_size,), "Incorrect label tensor shape"

    # -------------------------------------------------------------------------
    # 4. Initialize Model
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model...")

    model = AsymmetricEfficientNet(pretrained=True, dropout_rate=0.2)
    model = model.to(device)

    # Verification: Forward pass with dummy data
    with torch.no_grad():
        dummy_input = torch.randn(2, 12, 224, 224).to(device)
        dummy_output = model(dummy_input)
        print(f"Model Output Shape: {dummy_output.shape}")

    assert dummy_output.shape == (2, 1), "Model output should be (Batch, 1)"

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f} | Validation AUC: {val_auc:.4f}")

    # Assertions to ensure learning mechanics occurred (loss is a number)
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # Save demo model
    model_path = os.path.join(DEMO_DIR, "best_model_demo.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Predictions on Test Set...")

    model.eval()
    predictions = []
    ids = []

    # We iterate manually to pair IDs with predictions since the loader returns (image, dummy_label)
    # The dataset order is preserved because shuffle=False for test_loader

    # Get IDs from the dataframe used to create the loader
    test_ids = df_test_sub["BraTS21ID"].values

    all_probs = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)

    # Truncate or pad if necessary (though loader logic usually handles drop_last=False)
    # In this demo, sizes should match exactly.
    assert len(all_probs) == len(
        test_ids
    ), f"Mismatch: {len(all_probs)} preds vs {len(test_ids)} IDs"

    # Create submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    submission_path = os.path.join(DEMO_DIR, "submission_demo.csv")
    submission_df.to_csv(submission_path, index=False)

    print("Sample Submission Head:")
    print(submission_df.head())
    print(f"Submission saved to {submission_path}")

    print("\nDemo Execution Completed Successfully.")
