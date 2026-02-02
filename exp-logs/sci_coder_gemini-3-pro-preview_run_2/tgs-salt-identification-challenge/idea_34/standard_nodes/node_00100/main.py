import sys
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import library modules
from library.config import Config
from library.dataset import load_data, SaltDataset, get_transforms
from library.model import SaltModel
from library.losses import LovaszHingeLoss
from library.engine import (
    set_seed,
    fit_model,
    generate_submission,
    validate,
    do_kaggle_metric,
)
from library.utils import calculate_iou, optimize_threshold


def failure_analysis(model, val_loader, device):
    print("\n=== Failure Analysis ===")
    model.eval()

    ious = []
    depths_list = []
    salt_coverages = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks_gpu = masks.to(device, dtype=torch.float32)
            depths_gpu = depths.to(device, dtype=torch.float32)

            logits = model(images, depths_gpu)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Calculate IoU per image
            for i in range(len(images)):
                p = preds[i].cpu().numpy().flatten()
                t = masks_gpu[i].cpu().numpy().flatten()

                intersection = np.logical_and(p, t).sum()
                union = np.logical_or(p, t).sum()
                iou = intersection / union if union > 0 else 1.0

                ious.append(iou)
                depths_list.append(depths_gpu[i].item())
                salt_coverages.append(t.mean())

    ious = np.array(ious)
    depths_list = np.array(depths_list)
    salt_coverages = np.array(salt_coverages)

    # Correlations (Error = 1 - IoU)
    error = 1.0 - ious

    corr_depth = np.corrcoef(error, depths_list)[0, 1]
    if np.std(salt_coverages) > 0:
        corr_salt = np.corrcoef(error, salt_coverages)[0, 1]
    else:
        corr_salt = 0.0

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_salt:.4f}")

    return ious.mean()


def main():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_dataset = load_data("train", load_cached_data=True)
    val_dataset = load_data("val", load_cached_data=True)
    test_dataset = load_data("test", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Setup Model
    print("\n=== Training SaltModel (ResNet34 + Depth Injection) ===")
    model = SaltModel().to(device)

    # Loss: Lovasz + BCE
    bce_fn = nn.BCEWithLogitsLoss()
    lovasz_fn = LovaszHingeLoss()

    def criterion(logits, targets):
        return 0.5 * bce_fn(logits, targets) + 0.5 * lovasz_fn(logits, targets)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cite {solution_lesson_node_00068})
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 3. Train
    best_metric = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_fn=criterion,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
    )

    print(f"Best Val mAP: {best_metric}")

    # 4. Final Validation & Threshold Optimization
    print("\n=== Final Validation ===")
    model.load_state_dict(
        torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
    )
    model.eval()

    val_preds = []
    val_truths = []

    with torch.no_grad():
        for images, masks, depths, ids in val_loader:
            images = images.to(device, dtype=torch.float32)
            depths = depths.to(device, dtype=torch.float32)

            # Inference with TTA (Cite {solution_lesson_node_00095})
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            if Config.TTA_ENABLED:
                images_flip = torch.flip(images, [3])
                logits_flip = model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, [3])
                probs = (probs + probs_flip) / 2.0

            val_preds.append(probs.cpu().numpy())
            val_truths.append(masks.numpy())

    val_preds = np.concatenate(val_preds, axis=0).squeeze()
    val_truths = np.concatenate(val_truths, axis=0).squeeze()

    # Optimize Threshold
    best_threshold = optimize_threshold(val_preds, val_truths)
    print(f"Optimal Threshold: {best_threshold:.4f}")

    # Calculate Final Metric
    final_metric = do_kaggle_metric(val_preds, val_truths, threshold=best_threshold)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    failure_analysis(model, val_loader, device)

    # 6. Submission
    if final_metric > 0.7985:
        print("\nGenerating Submission...")
        generate_submission(model, test_loader, device, threshold=best_threshold)
    else:
        print(f"\nMetric {final_metric} <= 0.7985. Skipping submission.")


if __name__ == "__main__":
    main()
