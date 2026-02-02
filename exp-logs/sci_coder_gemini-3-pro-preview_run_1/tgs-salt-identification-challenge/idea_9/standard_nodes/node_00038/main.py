import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.stats import pearsonr

# Import from provided libraries
from library.utils import seed_everything, calculate_iou_map
from library.dataset import SaltDataset
from library.model import DeepResUNet
from library.loss import CompoundLoss
from library.trainer import train_one_epoch, validate
from library.inference import generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    seed_everything(42)

    # Paths
    WORK_DIR = "./working/idea_9"
    SAVE_DIR = os.path.join(WORK_DIR, "checkpoints")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Hyperparameters for Fast Baseline / Idea Validation
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 150  # 3 cycles of 50 epochs
    CYCLE_LEN = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading datasets...")
    train_dataset = SaltDataset(mode="train", work_dir=WORK_DIR, load_cached_data=True)
    val_dataset = SaltDataset(mode="val", work_dir=WORK_DIR, load_cached_data=True)

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

    # -------------------------------------------------------------------------
    # 3. Model, Loss, Optimizer
    # -------------------------------------------------------------------------
    print("Initializing model and optimizer...")
    model = DeepResUNet(in_channels=1, out_channels=1).to(device)
    criterion = CompoundLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cyclic Scheduler: T_0=50, T_mult=1 -> Restarts at 50, 100, 150
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=CYCLE_LEN, T_mult=1, eta_min=1e-6
    )

    # -------------------------------------------------------------------------
    # 4. Training Loop with Snapshot Logic
    # -------------------------------------------------------------------------
    print(f"Starting training for {EPOCHS} epochs...")

    best_map_cycle_2 = -1.0
    best_map_cycle_3 = -1.0

    path_cycle_2 = os.path.join(SAVE_DIR, "best_cycle_2.pth")
    path_cycle_3 = os.path.join(SAVE_DIR, "best_cycle_3.pth")

    start_time = time.time()

    for epoch in range(EPOCHS):
        # Check time limit (stop if > 1.8 hours to ensure time for eval/sub)
        if time.time() - start_time > 1.8 * 3600:
            print("Time limit approaching. Stopping training early.")
            break

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_map = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Snapshot Saving Logic
        # Cycle 2: Epochs 50 to 99 (indices)
        if 50 <= epoch < 100:
            if val_map > best_map_cycle_2:
                best_map_cycle_2 = val_map
                torch.save(model.state_dict(), path_cycle_2)

        # Cycle 3: Epochs 100 to 149 (indices)
        if 100 <= epoch < 150:
            if val_map > best_map_cycle_3:
                best_map_cycle_3 = val_map
                torch.save(model.state_dict(), path_cycle_3)

        # Logging (minimal)
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val mAP: {val_map:.4f}"
            )

    print("Training complete.")

    # -------------------------------------------------------------------------
    # 5. Ensemble Validation & Final Metric
    # -------------------------------------------------------------------------
    print("\n--- Performing Ensemble Validation ---")

    # Identify valid snapshots
    snapshots = []
    if os.path.exists(path_cycle_2):
        snapshots.append(path_cycle_2)
    if os.path.exists(path_cycle_3):
        snapshots.append(path_cycle_3)

    if not snapshots:
        print("No snapshots saved. Using current model state.")
        # Fallback: save current
        fallback_path = os.path.join(SAVE_DIR, "fallback.pth")
        torch.save(model.state_dict(), fallback_path)
        snapshots.append(fallback_path)

    # Load models
    models = []
    for p in snapshots:
        m = DeepResUNet(in_channels=1, out_channels=1).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)

    # Inference on Validation Set
    # We need to collect predictions and ground truths for metric calculation
    all_preds = []
    all_masks = []

    # Crop indices (128 -> 101)
    # Pad was 27 total (13, 14)
    start_idx = 13
    end_idx = 114

    with torch.no_grad():
        for images, masks, depths, _ in val_loader:
            images = images.to(device)
            depths = depths.to(device)

            # Ensemble Prediction
            batch_preds = 0.0
            for m in models:
                # Original
                out = torch.sigmoid(m(images, depths))
                batch_preds += out

                # TTA: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])
                out_flipped = torch.sigmoid(m(images_flipped, depths))
                batch_preds += torch.flip(out_flipped, dims=[3])

            # Average
            batch_preds /= len(models) * 2

            # Crop
            batch_preds = batch_preds[..., start_idx:end_idx, start_idx:end_idx]

            # Store
            all_preds.append(batch_preds.cpu())
            all_masks.append(
                masks.cpu()
            )  # Masks are already 101x101 in dataset, but padded in loader?
            # Wait, dataset returns padded masks. We must crop masks too.
            # Check dataset.py: __getitem__ returns padded masks (128x128).
            # So we must crop masks here as well.

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)  # (N, 1, 101, 101)
    all_masks_padded = torch.cat(all_masks, dim=0)  # (N, 1, 128, 128)

    # Crop masks to match preds
    all_masks_cropped = all_masks_padded[..., start_idx:end_idx, start_idx:end_idx]

    # Calculate Final Metric
    final_metric = calculate_iou_map(all_preds, all_masks_cropped)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # We need per-image scores. Re-implementing metric logic locally for per-sample analysis.
    # preds: (N, 1, 101, 101), masks: (N, 1, 101, 101)
    preds_np = all_preds.squeeze(1).numpy()
    masks_np = (all_masks_cropped.squeeze(1).numpy() > 0.5).astype(np.uint8)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    scores = []
    for i in range(len(preds_np)):
        p = (preds_np[i] > 0.5).astype(np.uint8)
        t = masks_np[i]

        t_sum = t.sum()
        p_sum = p.sum()

        if t_sum == 0:
            score = 1.0 if p_sum == 0 else 0.0
        else:
            if p_sum == 0:
                score = 0.0
            else:
                intersection = np.logical_and(t, p).sum()
                union = np.logical_or(t, p).sum()
                iou = intersection / union if union > 0 else 0.0
                matches = iou > iou_thresholds
                score = np.mean(matches)
        scores.append(score)

    scores = np.array(scores)
    errors = 1.0 - scores

    # Load metadata for analysis
    df_val = pd.read_csv("./metadata/val.csv")

    # Ensure alignment (loader doesn't shuffle)
    # Extract features
    depths = df_val["z"].values
    coverages = df_val["coverage"].values

    # Calculate image stats
    img_means = []
    img_stds = []

    # We need to reload images or rely on what we have?
    # We have processed 600 images in validation.
    # To be fast, let's just use the loaded dataset object if possible,
    # but dataset returns tensors.
    # Let's iterate the dataset directly (no padding) for stats to be accurate to original image
    # Or just use the metadata if available. Metadata doesn't have img_mean/std.
    # We will compute from the raw images in the dataset object.

    raw_images = val_dataset.images  # (N, 101, 101)
    for img in raw_images:
        img_means.append(np.mean(img))
        img_stds.append(np.std(img))

    img_means = np.array(img_means)
    img_stds = np.array(img_stds)

    # Correlations
    corr_depth, _ = pearsonr(errors, depths)
    corr_cov, _ = pearsonr(errors, coverages)
    corr_mean, _ = pearsonr(errors, img_means)
    corr_std, _ = pearsonr(errors, img_stds)

    print("Correlation between Error (1-mAP) and features:")
    print(f"  Depth (z): {corr_depth:.4f}")
    print(f"  Salt Coverage: {corr_cov:.4f}")
    print(f"  Image Mean Intensity: {corr_mean:.4f}")
    print(f"  Image Std Deviation: {corr_std:.4f}")

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.833:
        print("\nValidation metric meets threshold (> 0.833). Generating submission...")
        generate_submission(
            snapshot_paths=snapshots,
            work_dir=WORK_DIR,
            output_path="./submission/submission.csv",
            batch_size=BATCH_SIZE,
            device_name="cuda" if torch.cuda.is_available() else "cpu",
            load_cached_data=True,
        )
    else:
        print(
            f"\nValidation metric {final_metric:.4f} is below threshold 0.833. Submission skipped."
        )


if __name__ == "__main__":
    main()
