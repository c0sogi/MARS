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
from library.losses import SegmentationLoss
from library.dataset import get_dataloaders, SaltDataset
from library.engine import (
    set_seed,
    train_one_epoch,
    evaluate,
    predict,
    generate_submission,
)

# -------------------------------------------------------------------------
# Configuration Overrides
# -------------------------------------------------------------------------
Config.STAGE1_EPOCHS = (
    50  # Maximize supervised convergence (Cite solution_lesson_node_00045)
)
Config.BATCH_SIZE = 64
Config.NUM_WORKERS = 4


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Create working directories
    Config.setup()

    # -------------------------------------------------------------------------
    # Supervised Training
    # -------------------------------------------------------------------------
    print("\n=== Supervised Training (Depth Injection) ===")

    # Load Data (Disable cache loading to ensure fresh data if needed, or use optimized cache)
    # We use cache for speed, assuming data hasn't changed.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # Initialize Model
    model = ResNet34WideLinkNet(pretrained=True).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )
    criterion = SegmentationLoss()

    # Training Loop
    best_val_map = 0.0
    best_threshold = 0.5
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, Config.STAGE1_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.STAGE1_EPOCHS}")
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate with Adaptive Thresholding (Cite solution_lesson_node_00033)
        val_loss, val_map, val_thresh = evaluate(model, val_loader, criterion, device)

        if val_map > best_val_map:
            best_val_map = val_map
            best_threshold = val_thresh
            torch.save(model.state_dict(), best_model_path)
            print(
                f"  [Saved Best Model] Val mAP: {val_map:.4f} (Thresh: {val_thresh:.2f})"
            )

    # -------------------------------------------------------------------------
    # Evaluation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n=== Final Evaluation & Failure Analysis ===")

    # Load Best Model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 1. Final Validation Metric
    _, final_map, final_thresh = evaluate(model, val_loader, criterion, device)
    print(f"Final Validation Metric: {final_map}")

    # 2. Failure Analysis
    print("Performing Failure Analysis...")

    val_ious = []
    val_depths_list = []

    # Get depth stats for denormalization
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    depth_mean = train_df["z"].mean()
    depth_std = train_df["z"].std() + 1e-6

    with torch.no_grad():
        for batch in val_loader:
            images, masks, depths = batch
            images = images.to(device)
            depths = depths.to(device)

            # Predict
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            probs_np = probs.cpu().numpy()
            masks_np = masks.numpy()
            depths_np = depths.cpu().numpy()

            # Unpad and calc IoU per image
            for i in range(len(images)):
                p = unpad_image(probs_np[i, 0], Config.ORIG_H, Config.ORIG_W)
                m = unpad_image(masks_np[i, 0], Config.ORIG_H, Config.ORIG_W)

                # Binarize at optimal threshold
                p_bin = (p > final_thresh).astype(np.uint8)
                m_bin = (m > 0.5).astype(np.uint8)

                intersection = np.sum(p_bin & m_bin)
                union = np.sum(p_bin | m_bin)

                if union == 0:
                    iou = 1.0
                else:
                    iou = intersection / union

                val_ious.append(iou)

                # Denormalize depth
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

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    THRESHOLD_SCORE = 0.7985

    if final_map > THRESHOLD_SCORE:
        print(
            f"\nValidation mAP ({final_map:.4f}) > Threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        generate_submission(
            model,
            test_loader,
            device,
            output_path=Config.SUBMISSION_PATH,
            threshold=final_thresh,
        )
    else:
        print(
            f"\nValidation mAP ({final_map:.4f}) <= Threshold ({THRESHOLD_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
