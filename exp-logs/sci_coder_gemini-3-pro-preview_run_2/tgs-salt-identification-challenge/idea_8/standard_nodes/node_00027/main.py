import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    DEVICE,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    SUBMISSION_PATH,
)
from library.utils import set_seed, metric_map
from library.dataset import SaltDataset, get_depth_stats, get_transforms
from library.model import DepthRegularizedWideLinkNet
from library.losses import MixedLoss
from library.engine import train_one_epoch, validate, predict_test, crop_to_original


def main():
    # 1. Reproducibility
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Preparation
    print("Loading data...")
    # Calculate depth stats from training data for normalization
    depth_mean, depth_std = get_depth_stats(TRAIN_METADATA_PATH)
    depth_stats = (depth_mean, depth_std)

    # Initialize Datasets
    train_dataset = SaltDataset(
        TRAIN_METADATA_PATH,
        mode="train",
        depth_stats=depth_stats,
        transform=get_transforms("train"),
    )

    val_dataset = SaltDataset(
        VAL_METADATA_PATH,
        mode="val",
        depth_stats=depth_stats,
        transform=get_transforms("val"),
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = DepthRegularizedWideLinkNet(n_classes=1)
    model.to(DEVICE)

    # 4. Training Setup
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    loss_fn = MixedLoss(bce_weight=1.0, lovasz_weight=1.0)

    # 5. Training Loop
    best_score = -1.0
    best_threshold = 0.5
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)

        # Validate
        val_loss, score, threshold = validate(model, val_loader, loss_fn, DEVICE)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mAP: {score:.4f} | Threshold: {threshold:.2f}"
        )

        # Save Best
        if score > best_score:
            best_score = score
            best_threshold = threshold
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Validation mAP: {best_score:.4f}")

    # 6. Final Evaluation & Failure Analysis
    print("\n--- Final Evaluation & Failure Analysis ---")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # We need to compute per-image IoU for failure analysis.
    # We'll iterate through the validation set one more time.

    val_df = pd.read_csv(VAL_METADATA_PATH)
    all_ious = []

    with torch.no_grad():
        for images, masks, depths, _ in val_loader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)
            masks = masks.to(DEVICE)

            # Predict
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # Crop to original size
            probs = crop_to_original(probs)
            masks = crop_to_original(masks)

            # Binarize using the optimal threshold found during validation
            preds = (probs > best_threshold).float()

            # Calculate IoU per image in batch
            # Shape: (B, 1, H, W) -> (B, H*W)
            preds_flat = preds.view(preds.size(0), -1)
            masks_flat = masks.view(masks.size(0), -1)

            intersection = (preds_flat * masks_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1) - intersection

            # IoU = Int / Union. If Union == 0, IoU = 1.0 (Correct empty prediction)
            batch_ious = torch.ones_like(intersection)
            non_empty = union > 0
            batch_ious[non_empty] = intersection[non_empty] / union[non_empty]

            all_ious.append(batch_ious.cpu().numpy())

    all_ious = np.concatenate(all_ious)

    # Print Final Metric (Required Format)
    print(f"Final Validation Metric: {best_score}")

    # Failure Analysis Correlations
    # Error = 1 - IoU (at best threshold)
    errors = 1.0 - all_ious

    # Ensure metadata alignment
    if len(errors) != len(val_df):
        print(
            f"Warning: Validation set size mismatch. Preds: {len(errors)}, Metadata: {len(val_df)}"
        )
        n = min(len(errors), len(val_df))
        errors = errors[:n]
        val_df = val_df.iloc[:n]

    depths = val_df["z"].values
    coverages = val_df["salt_coverage"].values

    corr_depth = np.corrcoef(errors, depths)[0, 1]
    corr_coverage = np.corrcoef(errors, coverages)[0, 1]

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.4f}")

    # 7. Submission
    TARGET_METRIC = 0.7916666666666666

    if best_score > TARGET_METRIC:
        print(f"\nMetric {best_score} > {TARGET_METRIC}. Generating submission...")

        test_dataset = SaltDataset(
            TEST_METADATA_PATH,
            mode="test",
            depth_stats=depth_stats,
            transform=get_transforms("test"),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        predict_test(model, test_loader, best_threshold, DEVICE)
    else:
        print(f"\nMetric {best_score} <= {TARGET_METRIC}. Submission skipped.")


if __name__ == "__main__":
    main()
