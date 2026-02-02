import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import from library
from library.utils import set_seed, get_score
from library.model import SaltNet
from library.dataset import (
    preload_data,
    SaltDataset,
    get_transforms,
    get_depth_stats,
)
from library.losses import MixedLoss
from library.engine import (
    train_one_epoch,
    validate,
    optimize_threshold,
    generate_submission,
)

# Configuration
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METADATA_DIR = "./metadata"
CHECKPOINT_DIR = "./working/checkpoints"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
THRESHOLD_SCORE = 0.7985


def main():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    # 2. Data Loading
    print("Loading metadata and caching data...")
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    val_meta_path = os.path.join(METADATA_DIR, "val.csv")
    test_meta_path = os.path.join(METADATA_DIR, "test.csv")

    # Load Train Data
    df_train, data_train = preload_data(
        train_meta_path, phase="train", load_cached_data=True
    )
    # Load Val Data
    df_val, data_val = preload_data(val_meta_path, phase="val", load_cached_data=True)

    # Calculate Depth Stats from Train (used for normalization)
    # We reconstruct the dataframe logic for stats or just use the array
    # The dataset class expects a dict {'mean': ..., 'std': ...}
    depth_stats = {
        "mean": np.mean(data_train["depths"]),
        "std": np.std(data_train["depths"]),
    }
    print(f"Depth Stats: Mean={depth_stats['mean']:.4f}, Std={depth_stats['std']:.4f}")

    # Create Datasets
    train_dataset = SaltDataset(
        data_train, transform=get_transforms(phase="train"), depth_stats=depth_stats
    )
    val_dataset = SaltDataset(
        data_val, transform=get_transforms(phase="val"), depth_stats=depth_stats
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing SaltNet...")
    model = SaltNet().to(DEVICE)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # Loss Function (Mixed Lovasz + BCE)
    criterion = MixedLoss(alpha=1.0, beta=1.0).to(DEVICE)

    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs...")
    best_map = 0.0
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss, val_map = validate(model, val_loader, criterion, DEVICE)

        # Step Scheduler
        scheduler.step()

        # Checkpoint
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), best_model_path)
            # print(f"Epoch {epoch+1}: New Best mAP: {best_map:.4f}")

        # Logging (minimal)
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_map:.4f}"
            )

    print("Training complete.")

    # 5. Final Evaluation & Threshold Optimization
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    # Optimize Threshold
    best_threshold = optimize_threshold(model, val_loader, DEVICE)

    # Calculate Final Metric on Validation Set using optimized threshold
    # We need to run validate logic again but with the specific threshold
    # Since validate() in engine returns mAP with threshold=None (0.5 on logits),
    # we manually calculate it here to be precise and print it.

    # Re-run inference on validation to get predictions for final scoring and failure analysis
    print("Running final validation inference...")
    all_preds = []
    all_masks = []
    all_depths = []

    # We need access to original masks and depths for analysis
    # The loader returns transformed masks. We can use them or the raw data.
    # Using loader ensures alignment.

    with torch.no_grad():
        for images, masks, depths, _ in val_loader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            # Forward
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()  # Keep on CPU
            depths_np = depths.cpu().numpy()

            for i in range(probs_np.shape[0]):
                # Unpad
                from library.dataset import unpad_image

                p = unpad_image(probs_np[i, 0])
                m = unpad_image(masks_np[i, 0])

                all_preds.append(p)
                all_masks.append(m)
                all_depths.append(depths_np[i, 0])

    all_preds = np.array(all_preds)
    all_masks = (np.array(all_masks) > 0).astype(np.uint8)
    all_depths = np.array(all_depths)  # Normalized depths

    # Calculate Score
    final_metric = get_score(all_preds, all_masks, threshold_value=best_threshold)
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate per-image mAP approximation or IoU at best threshold
    # The metric is mean(AP over thresholds).
    # We will approximate "Error Magnitude" as 1.0 - IoU(at best_threshold)
    # to see which images failed the most.

    from library.utils import calc_iou_batch

    # Binarize preds
    binary_preds = (all_preds > best_threshold).astype(np.uint8)

    # Calculate IoU per image
    ious = calc_iou_batch(binary_preds, all_masks)
    errors = 1.0 - ious

    # Get original depths (un-normalize)
    # z_norm = (z - mean) / std => z = z_norm * std + mean
    orig_depths = all_depths * (depth_stats["std"] + 1e-8) + depth_stats["mean"]

    # Calculate Salt Coverage per image (GT)
    salt_coverage = np.mean(all_masks, axis=(1, 2))

    # Correlation: Error vs Depth
    corr_depth, _ = pearsonr(errors, orig_depths)
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")

    # Correlation: Error vs Salt Coverage
    corr_cov, _ = pearsonr(errors, salt_coverage)
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 7. Submission
    if final_metric > THRESHOLD_SCORE:
        print(f"\nMetric {final_metric} > {THRESHOLD_SCORE}. Generating submission...")

        # Load Test Data
        df_test, data_test = preload_data(
            test_meta_path, phase="test", load_cached_data=True
        )

        test_dataset = SaltDataset(
            data_test, transform=get_transforms(phase="test"), depth_stats=depth_stats
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Generate Submission
        # Using Marginalized Inference
        generate_submission(
            model,
            test_loader,
            DEVICE,
            output_path=SUBMISSION_PATH,
            threshold=best_threshold,
        )
    else:
        print(
            f"\nMetric {final_metric} <= {THRESHOLD_SCORE}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
