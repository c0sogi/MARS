import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from itertools import cycle

# Import library modules
from library.config import Config
from library.utils import (
    load_data_with_cache,
    unpad_image,
    calc_map_score,
    create_submission,
    rle_encode,
)
from library.model import ResNet34WideLinkNet
from library.losses import MultiTaskLoss
from library.dataset import get_dataloaders, SaltDataset
from library.engine import set_seed, train_one_epoch, evaluate, predict

# -------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -------------------------------------------------------------------------
# To ensure execution within 2 hours, we reduce epochs.
# A100 is fast, but we want to be safe.
Config.STAGE1_EPOCHS = 12
Config.STAGE3_EPOCHS = 12
Config.BATCH_SIZE = 64  # Increase batch size for A100
Config.NUM_WORKERS = 4


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Create working directories
    Config.setup()

    # -------------------------------------------------------------------------
    # Stage 1: Train Teacher Model (Supervised)
    # -------------------------------------------------------------------------
    print("\n=== Stage 1: Training Teacher Model ===")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Initialize Model
    teacher_model = ResNet34WideLinkNet(pretrained=True).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        teacher_model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    criterion = MultiTaskLoss(aux_weight=Config.AUX_DEPTH_LOSS_WEIGHT)

    # Scheduler (Cite solution_lesson_node_00035)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.STAGE1_EPOCHS
    )

    # Training Loop
    best_val_map = 0.0
    best_teacher_path = os.path.join(Config.CHECKPOINT_DIR, "teacher_best.pth")

    for epoch in range(1, Config.STAGE1_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.STAGE1_EPOCHS}")
        train_loss, train_metrics = train_one_epoch(
            teacher_model, train_loader, criterion, optimizer, device, epoch
        )

        scheduler.step()

        val_loss, val_map = evaluate(teacher_model, val_loader, criterion, device)

        # Save based on mAP (Cite solution_lesson_node_00033)
        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(teacher_model.state_dict(), best_teacher_path)
            print(
                f"  [Saved Best Teacher] Val Loss: {val_loss:.4f} | mAP: {val_map:.4f}"
            )

    # -------------------------------------------------------------------------
    # Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Final Evaluation & Failure Analysis ===")

    # Load Best Teacher (which is our main model now)
    teacher_model.load_state_dict(torch.load(best_teacher_path, map_location=device))
    teacher_model.eval()

    # 1. Final Validation Metric
    # We need to compute it exactly as requested
    _, final_map = evaluate(teacher_model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_map}")

    # 2. Failure Analysis
    # Calculate per-image IoU and correlate with Depth
    print("Performing Failure Analysis...")

    val_ious = []
    val_depths_list = []

    with torch.no_grad():
        for batch in val_loader:
            images, masks, depths = batch
            images = images.to(device)

            # Predict
            logits, _ = teacher_model(images, depths)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()
            depths_np = depths.numpy()

            # Unpad and calc IoU per image
            for i in range(len(images)):
                p = unpad_image(probs_np[i, 0], Config.ORIG_H, Config.ORIG_W)
                m = unpad_image(masks_np[i, 0], Config.ORIG_H, Config.ORIG_W)

                # Binarize at 0.5
                p_bin = (p > 0.5).astype(np.uint8)
                m_bin = (m > 0.5).astype(np.uint8)

                intersection = np.sum(p_bin & m_bin)
                union = np.sum(p_bin | m_bin)

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                val_ious.append(iou)

                # Denormalize depth for interpretation
                # depth_tensor was (z - mean) / std
                d_raw = (depths_np[i] * depth_std) + depth_mean
                val_depths_list.append(d_raw)

    val_ious = np.array(val_ious)
    val_depths_list = np.array(val_depths_list).flatten()

    # Calculate Error
    errors = 1.0 - val_ious

    # Correlation
    correlation = np.corrcoef(errors, val_depths_list)[0, 1]

    print(f"Failure Analysis Report:")
    print(f"  Mean IoU: {np.mean(val_ious):.4f}")
    print(f"  Mean Error: {np.mean(errors):.4f}")
    print(f"  Correlation (Error vs Depth): {correlation:.4f}")

    if abs(correlation) > 0.2:
        print(
            "  -> Significant correlation detected. Depth is a likely failure factor."
        )
    else:
        print("  -> Low correlation. Errors are likely depth-independent.")

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.7985

    if final_map > THRESHOLD_SCORE:
        print(
            f"\nValidation mAP ({final_map:.4f}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )

        # Generate prediction on Test Set
        # Use predict() which handles TTA and unpadding
        test_ids, test_probs = predict(teacher_model, test_loader, device)

        # Binarize
        binary_masks = (test_probs > 0.5).astype(np.uint8)

        # Save
        create_submission(test_ids, binary_masks, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nValidation mAP ({final_map:.4f}) <= Threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
