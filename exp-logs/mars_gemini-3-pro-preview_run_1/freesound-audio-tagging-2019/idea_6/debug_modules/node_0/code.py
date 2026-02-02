import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import from library
from library.config import Config
from library.utils import set_seed, calculate_lwlrap
from library.dataset import AudioDataset, collate_fn
from library.model import AudioClassifier
from library.augmentations import SpecAugment, mixup_data
from library.engine import fit


def run_demo():
    print("Initializing Demo...")

    # 1. Setup Environment and Config overrides for speed
    set_seed(42)

    # Override Config values for the demonstration to ensure it runs quickly
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.EPOCHS = 1

    print(f"Output Directory: {Config.OUTPUT_DIR}")

    # 2. Verify Metric Calculation (LWLRAP)
    print("\n--- Verifying Metric Calculation ---")
    # Create dummy ground truth and scores: 3 samples, 4 classes
    y_true = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
    y_score = np.array(
        [[0.8, 0.1, 0.1, 0.0], [0.2, 0.6, 0.5, 0.1], [0.1, 0.1, 0.1, 0.9]]
    )

    score = calculate_lwlrap(y_true, y_score)
    print(f"Calculated LWLRAP: {score:.4f}")

    # Basic assertions for metric correctness
    assert 0.0 <= score <= 1.0, "LWLRAP score out of range"
    perfect_score = calculate_lwlrap(y_true, y_true)
    assert np.isclose(perfect_score, 1.0), "Perfect score should be 1.0"

    # 3. Verify Dataset and DataLoader
    print("\n--- Verifying Dataset and DataLoader ---")
    # Initialize Datasets
    train_ds = AudioDataset(mode="train")
    val_ds = AudioDataset(mode="val")

    print(f"Full Train Dataset Size: {len(train_ds)}")
    print(f"Full Val Dataset Size: {len(val_ds)}")

    # Create Subsets for speed (use very small number of samples)
    train_subset = Subset(train_ds, list(range(12)))
    val_subset = Subset(val_ds, list(range(8)))

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    # Check one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["target"]
    fnames = batch["fname"]

    print(f"Batch Image Shape: {images.shape}")  # Expected: (B, 3, Freq, Time)
    print(f"Batch Target Shape: {targets.shape}")  # Expected: (B, Num_Classes)

    # Assertions for data shapes
    assert images.dim() == 4, "Images should be 4D (B, C, F, T)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert (
        targets.shape[1] == Config.NUM_CLASSES
    ), f"Targets should have {Config.NUM_CLASSES} classes"

    # 4. Verify Augmentations
    print("\n--- Verifying Augmentations ---")
    # SpecAugment
    spec_aug = SpecAugment()
    aug_images = spec_aug(images)
    print(f"Augmented Image Shape: {aug_images.shape}")
    assert aug_images.shape == images.shape, "SpecAugment changed tensor shape"

    # Mixup
    mixed_images, mixed_targets = mixup_data(images, targets, alpha=0.4)
    print(f"Mixed Image Shape: {mixed_images.shape}")
    assert mixed_images.shape == images.shape, "Mixup changed image shape"
    assert mixed_targets.shape == targets.shape, "Mixup changed target shape"

    # 5. Verify Model
    print("\n--- Verifying Model ---")
    # Initialize model (pretrained=False to avoid download time/errors in demo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AudioClassifier(pretrained=False).to(device)

    # Forward pass check
    with torch.no_grad():
        dummy_input = images.to(device)
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        images.shape[0],
        Config.NUM_CLASSES,
    ), "Output shape mismatch"

    # 6. Run Training Loop (Engine)
    print("\n--- Running Training Loop (Fit) ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Run fit for 1 epoch on the subset
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=1,  # Force 1 epoch
    )

    # 7. Inference on Test Subset
    print("\n--- Running Inference Demo ---")
    test_ds = AudioDataset(mode="test")
    test_subset = Subset(test_ds, list(range(4)))  # Just 4 samples
    test_loader = DataLoader(
        test_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()
    predictions = []
    file_names = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            fnames_batch = batch["fname"]

            out = model(imgs)
            probs = torch.sigmoid(out)

            predictions.append(probs.cpu().numpy())
            file_names.extend(fnames_batch)

    predictions = np.concatenate(predictions, axis=0)

    # Create submission DataFrame
    sub_df = pd.DataFrame(predictions, columns=test_ds.label_cols)
    sub_df.insert(0, "fname", file_names)

    print("Sample Prediction DataFrame:")
    print(sub_df.head())

    # Verify output format
    assert len(sub_df) == 4, "Prediction count mismatch"
    assert sub_df.shape[1] == Config.NUM_CLASSES + 1, "Column count mismatch"

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
