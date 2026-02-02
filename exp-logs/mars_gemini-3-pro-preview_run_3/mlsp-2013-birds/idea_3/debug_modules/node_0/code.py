import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mixup_data, calculate_metric
from library.dataset import get_dataloaders, BirdDataset
from library.model import BirdResNet, AttentionPooling
from library.trainer import train_one_epoch, validate_one_epoch, generate_submission


def main():
    print("==== Starting Library Demonstration ====")

    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Ensure working directory exists (Config.setup() does this on import, but good to double check logic)
    assert os.path.exists(Config.WORKING_DIR), "Working directory should exist."

    # 2. Dataset and DataLoader Demonstration
    print("\n[Dataset] Initializing DataLoaders in debug mode...")
    # debug=True loads only 50 samples for speed
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    print("[Dataset] Fetching one batch from Train Loader...")
    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Verify shapes
    # Expected: (Batch_Size, 3, N_MELS, TimeSteps)
    # TimeSteps depends on DURATION, SR, HOP_LENGTH.
    # 10s * 16000 / 320 = 500 frames + 1 = 501.
    expected_time_steps = int(Config.SR * Config.DURATION / Config.HOP_LENGTH) + 1

    print(f"  Image Batch Shape: {images.shape}")
    print(f"  Label Batch Shape: {labels.shape}")

    assert images.dim() == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (replicated mono)"
    assert (
        images.shape[2] == Config.N_MELS
    ), f"Height should be N_MELS ({Config.N_MELS})"
    assert (
        images.shape[3] == expected_time_steps
    ), f"Width should be {expected_time_steps}"
    assert (
        labels.shape[1] == Config.NUM_CLASSES
    ), f"Labels should have {Config.NUM_CLASSES} classes"

    # Verify Normalization (roughly, since it's ImageNet normalized)
    print(f"  Batch Mean: {images.mean().item():.4f}, Std: {images.std().item():.4f}")

    # 3. Model Demonstration
    print("\n[Model] Instantiating BirdResNet...")
    model = BirdResNet(pretrained=False, num_classes=Config.NUM_CLASSES)
    model = model.to(device)

    # Test Attention Pooling specifically
    print("[Model] Verifying AttentionPooling layer...")
    feat_dim = 512
    time_dim = 50
    dummy_features = torch.randn(2, feat_dim, time_dim).to(
        device
    )  # (Batch, Channels, Time)
    att_layer = AttentionPooling(in_channels=feat_dim).to(device)
    pooled = att_layer(dummy_features)

    assert pooled.shape == (
        2,
        feat_dim,
    ), f"Attention pooling output shape mismatch. Got {pooled.shape}"
    print("  AttentionPooling check passed.")

    # Test Full Forward Pass
    print("[Model] Running forward pass on real batch...")
    images = images.to(device)
    logits = model(images)

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Logits shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"
    print("  Forward pass successful.")

    # 4. Utils Demonstration
    print("\n[Utils] Testing Mixup...")
    mixed_x, y_a, y_b, lam = mixup_data(
        images, labels.to(device), alpha=0.4, device=device
    )
    assert mixed_x.shape == images.shape, "Mixed images shape mismatch"
    assert y_a.shape == labels.shape, "Target A shape mismatch"
    assert 0 <= lam <= 1, "Lambda should be between 0 and 1"
    print("  Mixup check passed.")

    print("[Utils] Testing Metric Calculation...")
    # Create synthetic data for metric test
    # 2 samples, 19 classes.
    # Class 0: True=[0, 1], Pred=[0.1, 0.9] -> AUC 1.0
    y_true_syn = np.zeros((2, Config.NUM_CLASSES))
    y_true_syn[0, 0] = 0
    y_true_syn[1, 0] = 1

    y_pred_syn = np.zeros((2, Config.NUM_CLASSES))
    y_pred_syn[0, 0] = 0.1
    y_pred_syn[1, 0] = 0.9

    # Other classes are all 0, so calculate_metric should handle undefined AUCs gracefully (skip them)
    # and return the average of valid AUCs (which is just Class 0 here)
    score = calculate_metric(y_true_syn, y_pred_syn)
    print(f"  Calculated AUC: {score}")
    assert score == 1.0, "Metric calculation failed for simple case."

    # 5. Training Loop Simulation
    print("\n[Trainer] Simulating Training Loop (1 Epoch)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"  Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate one epoch
    val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)
    print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Submission Generation
    print("\n[Trainer] Generating Submission...")
    # Define a temporary output path for this test
    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(model, test_loader, device, demo_submission_path)

    assert os.path.exists(demo_submission_path), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(demo_submission_path)
    print(f"  Submission rows: {len(df_sub)}")
    print(f"  Submission columns: {df_sub.columns.tolist()}")

    # In debug mode, test_loader has Config.DEBUG_SUBSET_SIZE samples (50).
    # Each sample has 19 classes.
    # Total rows should be 50 * 19 = 950.
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.NUM_CLASSES
    # However, if the test set is smaller than DEBUG_SUBSET_SIZE, it will be smaller.
    # The actual test set is 64 samples. 50 < 64, so we expect 950 rows.

    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns mismatch."

    print("\n==== Demonstration Complete ====")


if __name__ == "__main__":
    main()
