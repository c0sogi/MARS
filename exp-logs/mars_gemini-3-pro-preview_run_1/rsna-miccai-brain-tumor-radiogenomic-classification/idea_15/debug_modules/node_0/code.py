import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import (
    TRAIN_METADATA_PATH,
    TEST_METADATA_PATH,
    DEVICE,
    BATCH_SIZE,
    WORKING_DIR,
    SEED,
)
from library.utils import seed_everything
from library.dataset import SlabDataset, get_transforms
from library.model import WITSNetwork
from library.train import train_one_epoch


def main():
    print("Starting WITS-II Pipeline Demonstration...")

    # 1. Reproducibility
    seed_everything(SEED)

    # ==========================================
    # Part 1: Dataset Instantiation & Verification
    # ==========================================
    print("\n[1/4] Verifying Dataset Logic...")

    # Load metadata
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {TRAIN_METADATA_PATH}")

    full_train_df = pd.read_csv(TRAIN_METADATA_PATH)

    # Create a mini subset (5 subjects) for speed
    mini_train_df = full_train_df.head(5).copy()
    print(f"Created mini training set with {len(mini_train_df)} subjects.")

    # Instantiate Dataset
    # We use a unique split_name to generate a separate cache file in ./working
    dataset = SlabDataset(
        metadata_df=mini_train_df,
        transform=get_transforms("train"),
        load_cached_data=False,  # Force processing to verify logic
        split_name="demo_train",
    )

    # Verification 1: Length
    # Each subject produces 3 slabs (NUM_SLABS=3 in config)
    expected_len = len(mini_train_df) * 3
    assert (
        len(dataset) == expected_len
    ), f"Dataset length mismatch. Expected {expected_len}, got {len(dataset)}"
    print(f"Dataset length verified: {len(dataset)} samples.")

    # Verification 2: Item Shape
    # Shape should be (Channels, H, W) -> (9, 224, 224)
    sample_img, sample_target = dataset[0]
    assert sample_img.shape == (
        9,
        224,
        224,
    ), f"Image shape mismatch. Expected (9, 224, 224), got {sample_img.shape}"
    assert isinstance(sample_target, torch.Tensor), "Target is not a tensor"
    print(f"Sample shape verified: {sample_img.shape}")

    # ==========================================
    # Part 2: Model Initialization & Forward Pass
    # ==========================================
    print("\n[2/4] Verifying Model Architecture...")

    model = WITSNetwork()
    model.to(DEVICE)

    # Create a DataLoader
    # Use a small batch size for the demo
    demo_batch_size = 4
    loader = DataLoader(dataset, batch_size=demo_batch_size, shuffle=True)

    # Get one batch
    images, targets = next(iter(loader))
    images = images.to(DEVICE)

    # Forward pass
    logits = model(images)

    # Verification 3: Output Shape
    # Should be (Batch_Size, 1)
    assert logits.shape == (
        demo_batch_size,
        1,
    ), f"Logits shape mismatch. Expected ({demo_batch_size}, 1), got {logits.shape}"
    print(f"Model forward pass successful. Output shape: {logits.shape}")

    # ==========================================
    # Part 3: Training Loop Demonstration
    # ==========================================
    print("\n[3/4] Demonstrating Training Step...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch using the library function
    loss, auc = train_one_epoch(model, loader, optimizer, criterion, DEVICE)

    # Verification 4: Loss validity
    assert not np.isnan(loss), "Training loss is NaN"
    assert loss > 0, "Training loss should be positive"
    print(f"Training step complete. Loss: {loss:.4f}, AUC: {auc:.4f}")

    # ==========================================
    # Part 4: Inference & Submission Logic
    # ==========================================
    print("\n[4/4] Demonstrating Inference & Aggregation...")

    # Load Test Metadata
    if not os.path.exists(TEST_METADATA_PATH):
        raise FileNotFoundError(f"Test metadata not found at {TEST_METADATA_PATH}")

    full_test_df = pd.read_csv(TEST_METADATA_PATH)
    mini_test_df = full_test_df.head(3).copy()  # 3 subjects

    # Instantiate Test Dataset
    test_dataset = SlabDataset(
        metadata_df=mini_test_df,
        transform=get_transforms("val"),
        load_cached_data=False,
        split_name="demo_test",
    )

    test_loader = DataLoader(test_dataset, batch_size=demo_batch_size, shuffle=False)

    # Run Inference
    model.eval()
    all_probs = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    # Flatten predictions
    flat_probs = np.concatenate(all_probs).flatten()

    # Verification 5: Prediction Count
    assert len(flat_probs) == len(
        test_dataset
    ), f"Prediction count mismatch. Expected {len(test_dataset)}, got {len(flat_probs)}"

    # Aggregation Logic (Mean of 3 slabs per subject)
    # The dataset stores IDs corresponding to each slab
    slab_ids = test_dataset.ids

    df_preds = pd.DataFrame({"BraTS21ID": slab_ids, "prob": flat_probs})
    submission_df = df_preds.groupby("BraTS21ID")["prob"].mean().reset_index()
    submission_df.rename(columns={"prob": "MGMT_value"}, inplace=True)

    # Verification 6: Submission Structure
    expected_subjects = mini_test_df["BraTS21ID"].nunique()
    assert (
        len(submission_df) == expected_subjects
    ), f"Submission row count mismatch. Expected {expected_subjects}, got {len(submission_df)}"
    assert (
        "BraTS21ID" in submission_df.columns and "MGMT_value" in submission_df.columns
    ), "Submission columns missing"

    print("Inference aggregation successful.")
    print("Sample Submission Output:")
    print(submission_df.to_string(index=False))

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
