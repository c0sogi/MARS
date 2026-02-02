import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import DeepResUNet
from library.losses import DeepSupervisionLoss
from library.metrics import calculate_iou_map
from library.predict import generate_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Override Config for Fast Baseline
    # 150 epochs with Cosine Annealing for robust convergence
    Config.EPOCHS = 150
    Config.DEBUG = False

    # 2. Prepare Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        debug=Config.DEBUG,
        load_cached_data=True,
    )

    # 3. Initialize Model
    model = DeepResUNet(
        in_channels=2, out_channels=1, deep_supervision=Config.DEEP_SUPERVISION
    )
    model = model.to(device)

    # 4. Optimization
    criterion = DeepSupervisionLoss(weights=Config.DS_WEIGHTS)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.T_0, T_mult=Config.T_MULT, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_map = 0.0

    # Crop slice for metric calculation (128 -> 101)
    pad_offset = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    crop_slice = slice(pad_offset, pad_offset + Config.ORIG_SIZE)

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass (returns list if deep supervision is on)
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation Step
        model.eval()
        val_loss_accum = 0.0
        val_map_accum = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass (returns single tensor in eval mode)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss_accum += loss.item()

                # Metric Calculation
                probs = torch.sigmoid(outputs)
                probs_cropped = probs[:, :, crop_slice, crop_slice]
                masks_cropped = masks[:, :, crop_slice, crop_slice]

                batch_map = calculate_iou_map(probs_cropped, masks_cropped)
                val_map_accum += batch_map

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_map = val_map_accum / len(val_loader)

        # Scheduler Step
        scheduler.step()

        # Save Best Model
        if avg_val_map > best_map:
            best_map = avg_val_map
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

    # 6. Final Validation & Failure Analysis
    print("Training complete. Loading best model for analysis...")
    if os.path.exists(Config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(Config.CHECKPOINT_PATH, map_location=device))

    model.eval()

    val_ious = []
    val_depths_list = []
    val_coverages_list = []

    # Access raw depths from dataset.
    # val_loader has shuffle=False, so order matches dataset.depths
    all_depths = val_loader.dataset.depths
    current_idx = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)

            # Forward
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # Crop to original size
            probs_cropped = probs[:, :, crop_slice, crop_slice].cpu().numpy()
            masks_cropped = masks[:, :, crop_slice, crop_slice].cpu().numpy()

            batch_size = images.size(0)

            for i in range(batch_size):
                # Get single sample data
                pred_mask = probs_cropped[i]  # (1, 101, 101)
                true_mask = masks_cropped[i]  # (1, 101, 101)

                # Calculate IoU/mAP for this single image
                score = calculate_iou_map(
                    pred_mask[np.newaxis, ...], true_mask[np.newaxis, ...]
                )
                val_ious.append(score)

                # Get metadata
                depth = all_depths[current_idx]
                val_depths_list.append(depth)

                # Calculate coverage (ratio of salt pixels)
                coverage = np.sum(true_mask) / true_mask.size
                val_coverages_list.append(coverage)

                current_idx += 1

    # Calculate Final Metric
    final_metric = np.mean(val_ious)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    # Error = 1 - mAP
    errors = 1.0 - np.array(val_ious)
    depths_arr = np.array(val_depths_list)
    coverages_arr = np.array(val_coverages_list)

    # Calculate Pearson Correlation
    if np.std(errors) > 0 and np.std(depths_arr) > 0:
        corr_depth = np.corrcoef(errors, depths_arr)[0, 1]
    else:
        corr_depth = 0.0

    if np.std(errors) > 0 and np.std(coverages_arr) > 0:
        corr_cov = np.corrcoef(errors, coverages_arr)[0, 1]
    else:
        corr_cov = 0.0

    print("Failure Analysis Results:")
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    # 7. Submission
    THRESHOLD = 0.8123333333333332
    if final_metric > THRESHOLD:
        print(f"Metric {final_metric} > {THRESHOLD}. Generating submission...")
        # Use the provided predict library function which handles TTA and formatting
        generate_submission(debug=Config.DEBUG)
    else:
        print(f"Metric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
