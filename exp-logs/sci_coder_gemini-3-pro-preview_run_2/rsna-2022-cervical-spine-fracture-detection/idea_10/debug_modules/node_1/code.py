import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_weighted_log_loss_score, AverageMeter
from library.data import CervicalSpineDataset, get_transforms, MetadataProcessor
from library.model import CervicalFractureNet
from library.loss import CervicalSpineLoss
from library.train import train_one_epoch, validate


def create_mini_metadata(original_path, output_path, n_samples=4):
    """Creates a small metadata file for demonstration purposes."""
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original metadata not found: {original_path}")

    df = pd.read_csv(original_path)
    # Take top n_samples
    mini_df = df.head(n_samples).copy()
    mini_df.to_csv(output_path, index=False)
    print(f"Created mini metadata at {output_path} with {len(mini_df)} samples.")
    return mini_df


def main():
    print("=== Starting Cervical Spine Fracture Detection Demo ===")

    # 1. Setup & Configuration Overrides for Speed
    print("\n[1] Configuring environment...")

    # Override Config for speed and lightweight execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.SEQ_LENGTH = 32  # Reduced from 96
    Config.BACKBONE = "resnet18"  # Lightweight backbone
    Config.FEATURE_DIM = 512  # ResNet18 final feature dim
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.ACCUMULATION_STEPS = 1
    Config.DEBUG = True

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seed
    set_seed(Config.SEED)

    # 2. Prepare Mini Datasets
    print("\n[2] Preparing mini datasets...")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")

    create_mini_metadata(Config.TRAIN_METADATA_PATH, mini_train_path, n_samples=4)
    create_mini_metadata(Config.VAL_METADATA_PATH, mini_val_path, n_samples=2)

    # Update Config paths to point to mini datasets
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path

    # 3. Data Loading Verification
    print("\n[3] Verifying Data Loading...")

    # Initialize MetadataProcessor to cache paths (using the mini metadata)
    proc = MetadataProcessor(Config.WORKING_DIR)

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Get paths (this will use the cache logic in library.data)
    # We construct a combined DF just for the path processor as the library does
    all_meta = pd.concat([train_df, val_df], ignore_index=True).drop_duplicates(
        subset=["StudyInstanceUID"]
    )
    path_map = proc.get_image_paths(all_meta, load_cached=False)
    bbox_map = proc.get_bbox_map()

    # Instantiate Dataset
    train_ds = CervicalSpineDataset(
        train_df, path_map, bbox_map, transform=get_transforms("train"), phase="train"
    )

    # Check __getitem__
    sample = train_ds[0]
    print(f"Sample keys: {sample.keys()}")

    # Assertions for shapes
    # Image: (Seq, 3, H, W)
    assert sample["image"].shape == (
        Config.SEQ_LENGTH,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch: {sample['image'].shape}"
    # Study Labels: (8,)
    assert sample["study_labels"].shape == (
        8,
    ), f"Study labels shape mismatch: {sample['study_labels'].shape}"
    # Slice Fracture Labels: (Seq,)
    assert sample["slice_fracture_labels"].shape == (
        Config.SEQ_LENGTH,
    ), f"Slice labels shape mismatch: {sample['slice_fracture_labels'].shape}"

    print("Dataset verification passed.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_ds = CervicalSpineDataset(
        val_df, path_map, bbox_map, transform=get_transforms("val"), phase="val"
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 4. Model Verification
    print("\n[4] Verifying Model Architecture...")
    model = CervicalFractureNet()
    model.to(device)

    # Create a dummy batch
    dummy_input = torch.randn(
        Config.BATCH_SIZE, Config.SEQ_LENGTH, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)

    # Forward pass
    outputs = model(dummy_input)

    # Check outputs
    study_logits = outputs["study_logits"]
    slice_logits = outputs["slice_fracture_logits"]
    spatial_maps = outputs["spatial_maps"]
    anatomy_logits = outputs["anatomy_logits"]

    print(f"Study Logits Shape: {study_logits.shape}")
    print(f"Slice Logits Shape: {slice_logits.shape}")

    assert study_logits.shape == (Config.BATCH_SIZE, 8), "Study logits shape incorrect"
    assert slice_logits.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        1,
    ), "Slice logits shape incorrect"
    assert anatomy_logits.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        8,
    ), "Anatomy logits shape incorrect"

    print("Model forward pass verification passed.")

    # 5. Loss Function Verification
    print("\n[5] Verifying Loss Function...")
    criterion = CervicalSpineLoss()

    # Create dummy targets
    dummy_targets = {
        "study_labels": torch.randint(0, 2, (Config.BATCH_SIZE, 8)).float().to(device),
        "slice_fracture_labels": torch.randint(
            0, 2, (Config.BATCH_SIZE, Config.SEQ_LENGTH)
        )
        .float()
        .to(device),
        "spatial_masks": torch.randint(
            0,
            2,
            (
                Config.BATCH_SIZE,
                Config.SEQ_LENGTH,
                Config.IMAGE_SIZE,
                Config.IMAGE_SIZE,
            ),
        )
        .float()
        .to(device),
        "anatomy_labels": torch.randint(0, 8, (Config.BATCH_SIZE, Config.SEQ_LENGTH))
        .long()
        .to(device),
        "has_bbox": torch.ones(Config.BATCH_SIZE).bool().to(device),
        "has_segmentation": torch.ones(Config.BATCH_SIZE).bool().to(device),
    }

    loss, metrics = criterion(outputs, dummy_targets)
    print(f"Calculated Loss: {loss.item()}")
    print(f"Metrics: {metrics}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Loss verification passed.")

    # 6. Training Loop Demonstration
    print("\n[6] Running Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()

    avg_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, scaler, device, epoch=1
    )
    print(f"Epoch 1 Average Loss: {avg_loss:.4f}")

    # 7. Validation Loop Demonstration
    print("\n[7] Running Validation Loop...")
    val_loss, val_metric = validate(val_loader, model, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Metric (Weighted Log Loss): {val_metric:.4f}")

    # 8. Metric Logic Verification
    print("\n[8] Verifying Metric Calculation Logic...")
    # Test case: Perfect prediction
    y_true = np.array([[0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 1]])
    y_pred_perfect = np.array(
        [[0.01] * 7 + [0.01], [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.99]]
    )

    # Weights are [1, 1, 1, 1, 1, 1, 1, 7]
    # Row 1: all correct negatives. Loss approx -log(0.99) * weight
    # Row 2: C1 correct pos, others correct neg, overall correct pos.

    score = get_weighted_log_loss_score(y_true, y_pred_perfect)
    print(f"Perfect Prediction Score (should be low): {score:.6f}")
    assert score < 0.2, "Metric score for good predictions is too high"

    # Test case: Worst prediction
    y_pred_bad = 1.0 - y_pred_perfect
    score_bad = get_weighted_log_loss_score(y_true, y_pred_bad)
    print(f"Bad Prediction Score (should be high): {score_bad:.6f}")
    assert score_bad > 2.0, "Metric score for bad predictions is too low"

    print("Metric logic verification passed.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
