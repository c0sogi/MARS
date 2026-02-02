import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from library
from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    calc_map,
    save_checkpoint,
    rle_encode,
    pad_image,
)
from library.dataset import SaltDataset
from library.model import SaltSegModel
from library.losses import BCEDiceLoss, LovaszHingeLoss, DeepSupervisionLoss
from library.engine import train_one_epoch, validate, generate_submission, predict_tta

# -------------------------------------------------------------------------
# Configuration Overrides for Execution
# -------------------------------------------------------------------------
# Adjusting epochs to fit within the 2-hour fast baseline requirement
# while maintaining sufficient convergence for the high score threshold.
# On A100, 50 epochs * 5 folds is approx 45-60 minutes.
Config.EPOCHS = 50
Config.LOVASZ_SWITCH_EPOCH = 15
Config.BATCH_SIZE = 64


def get_val_preds(model, loader, device):
    """
    Runs inference on validation set and returns raw probabilities, targets, and IDs.
    Used for OOF analysis and Threshold Optimization.
    """
    model.eval()
    all_probs = []
    all_targets = []
    all_ids = []

    # Padding offsets for cropping back to 101x101
    diff = Config.IMG_SIZE - Config.ORIG_SIZE
    pad_top = diff // 2
    pad_left = diff // 2

    with torch.no_grad():
        for images, masks, ids in loader:
            images = images.to(device, dtype=torch.float32)

            # Forward pass (Mixed Precision not strictly needed for inference but consistent)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(images)

            # Handle Deep Supervision
            if isinstance(outputs, (list, tuple)):
                logits = outputs[-1]
            else:
                logits = outputs

            probs = torch.sigmoid(logits)

            # Crop to original size
            probs_cropped = probs[
                :,
                :,
                pad_top : pad_top + Config.ORIG_SIZE,
                pad_left : pad_left + Config.ORIG_SIZE,
            ]
            masks_cropped = masks[
                :,
                :,
                pad_top : pad_top + Config.ORIG_SIZE,
                pad_left : pad_left + Config.ORIG_SIZE,
            ]

            # Store
            all_probs.append(probs_cropped.cpu().numpy())
            all_targets.append(masks_cropped.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs, axis=0).squeeze(1)
        all_targets = np.concatenate(all_targets, axis=0).squeeze(1)

    return all_probs, all_targets, np.array(all_ids)


def run_fold(fold_idx):
    print(f"\n{'='*20} Starting Fold {fold_idx+1}/{Config.FOLDS} {'='*20}")

    # 1. Data Loaders
    train_dataset = SaltDataset(mode="train", fold_index=fold_idx, n_folds=Config.FOLDS)
    val_dataset = SaltDataset(mode="val", fold_index=fold_idx, n_folds=Config.FOLDS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model & Optimizer
    model = SaltSegModel(encoder_name=Config.ENCODER_NAME, pretrained=True)
    model.to(Config.DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(Config.DEVICE == "cuda"))

    # 3. Losses
    # Warmup: BCE + Dice on all deep supervision heads
    criterion_warmup = DeepSupervisionLoss(BCEDiceLoss(), weights=[0.1, 0.1, 0.1, 1.0])
    # Fine-tune: Lovasz-Hinge on all deep supervision heads
    criterion_finetune = DeepSupervisionLoss(
        LovaszHingeLoss(per_image=True), weights=[0.1, 0.1, 0.1, 1.0]
    )

    best_map = 0.0
    best_epoch = 0

    # 4. Training Loop
    for epoch in range(1, Config.EPOCHS + 1):
        # Select Loss
        if epoch <= Config.LOVASZ_SWITCH_EPOCH:
            criterion = criterion_warmup
            loss_name = "BCE+Dice"
        else:
            criterion = criterion_finetune
            loss_name = "Lovasz"

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, Config.DEVICE, epoch
        )

        # Validate
        val_loss, val_map = validate(model, val_loader, criterion, Config.DEVICE)

        # Scheduler Step
        scheduler.step(val_map)

        print(
            f"Epoch {epoch} [{loss_name}]: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val mAP {val_map:.4f}"
        )

        # Save Best
        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_map": best_map,
                },
                is_best=True,
                filename=os.path.join(
                    Config.WORK_DIR, f"fold_{fold_idx}_checkpoint.pth"
                ),
            )

    print(f"Fold {fold_idx} Best mAP: {best_map:.4f} at Epoch {best_epoch}")

    # 5. Load Best Model for OOF Generation
    best_path = os.path.join(Config.WORK_DIR, f"fold_{fold_idx}_best.pth")
    checkpoint = torch.load(best_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    # Generate OOF Predictions
    probs, targets, ids = get_val_preds(model, val_loader, Config.DEVICE)

    # Clear memory
    del model, optimizer, scaler, train_loader, val_loader
    torch.cuda.empty_cache()

    return probs, targets, ids


def analyze_failures(probs, targets, ids, metadata_df):
    """
    Performs failure analysis correlating error with depth and salt coverage.
    """
    print("\n" + "=" * 20 + " FAILURE ANALYSIS " + "=" * 20)

    # Calculate per-image mAP approximation (using best threshold 0.5 for analysis)
    # Note: calc_map computes average over thresholds. We can't easily get per-image mAP
    # from the batch function without re-implementing.
    # Instead, we'll calculate IoU at 0.5 as a proxy for performance quality.

    preds_bin = (probs > 0.5).astype(np.uint8)
    targets_bin = targets.astype(np.uint8)

    ious = []
    for i in range(len(preds_bin)):
        p = preds_bin[i]
        t = targets_bin[i]

        if np.sum(t) == 0:
            iou = 1.0 if np.sum(p) == 0 else 0.0
        else:
            intersection = np.sum(p & t)
            union = np.sum(p | t)
            iou = intersection / (union + 1e-7)
        ious.append(iou)

    ious = np.array(ious)
    errors = 1.0 - ious

    # Map IDs to metadata
    meta_subset = metadata_df[metadata_df["id"].isin(ids)].set_index("id")
    # Reorder metadata to match ids array order
    meta_ordered = meta_subset.reindex(ids)

    depths = meta_ordered["z"].values
    coverages = meta_ordered["coverage"].values

    # Correlations
    # Handle NaNs if any (shouldn't be based on previous analysis)
    valid_mask = ~np.isnan(errors)

    corr_depth = np.corrcoef(errors[valid_mask], depths[valid_mask])[0, 1]
    corr_coverage = np.corrcoef(errors[valid_mask], coverages[valid_mask])[0, 1]

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_coverage:.4f}")

    return errors


def main():
    seed_everything(Config.SEED)

    # Load Metadata for Analysis
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    # Containers for OOF
    oof_probs = []
    oof_targets = []
    oof_ids = []

    # Run Folds
    for fold in range(Config.FOLDS):
        p, t, i = run_fold(fold)
        oof_probs.append(p)
        oof_targets.append(t)
        oof_ids.append(i)

    # Concatenate all OOF
    oof_probs = np.concatenate(oof_probs, axis=0)
    oof_targets = np.concatenate(oof_targets, axis=0)
    oof_ids = np.concatenate(oof_ids, axis=0)

    # -------------------------------------------------------------------------
    # Global Threshold Optimization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 20 + " THRESHOLD OPTIMIZATION " + "=" * 20)
    thresholds = np.arange(0.3, 0.75, 0.05)
    best_threshold = 0.5
    best_global_map = 0.0

    for t in thresholds:
        score = calc_map(oof_probs, oof_targets, threshold=t)
        print(f"Threshold {t:.2f}: mAP = {score:.5f}")
        if score > best_global_map:
            best_global_map = score
            best_threshold = t

    print(f"Best Threshold: {best_threshold:.2f} with mAP: {best_global_map:.5f}")

    # Print Required Metric Format
    print(f"Final Validation Metric: {best_global_map}")

    # -------------------------------------------------------------------------
    # Failure Analysis
    # -------------------------------------------------------------------------
    analyze_failures(oof_probs, oof_targets, oof_ids, full_meta)

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    if best_global_map > 0.827:
        print("\n" + "=" * 20 + " GENERATING SUBMISSION " + "=" * 20)

        # Load Test Data
        test_dataset = SaltDataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load all models
        models = []
        for fold in range(Config.FOLDS):
            model_path = os.path.join(Config.WORK_DIR, f"fold_{fold}_best.pth")
            model = SaltSegModel(encoder_name=Config.ENCODER_NAME, pretrained=False)
            checkpoint = torch.load(model_path, map_location=Config.DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(Config.DEVICE)
            model.eval()
            models.append(model)

        # Prediction Loop (Ensemble + TTA)
        submission_ids = []
        submission_rles = []

        # Padding offsets
        diff = Config.IMG_SIZE - Config.ORIG_SIZE
        pad_top = diff // 2
        pad_left = diff // 2

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(Config.DEVICE, dtype=torch.float32)

                # Ensemble Prediction
                batch_probs_sum = torch.zeros(
                    (images.size(0), 1, Config.IMG_SIZE, Config.IMG_SIZE),
                    device=Config.DEVICE,
                )

                for model in models:
                    # TTA Prediction per model
                    probs = predict_tta(model, images)
                    batch_probs_sum += probs

                # Average
                batch_avg_probs = batch_probs_sum / len(models)

                # Unpad
                probs_cropped = batch_avg_probs[
                    :,
                    :,
                    pad_top : pad_top + Config.ORIG_SIZE,
                    pad_left : pad_left + Config.ORIG_SIZE,
                ]
                probs_np = probs_cropped.cpu().numpy().squeeze(1)

                # Process Batch
                for i in range(len(ids)):
                    img_id = ids[i]
                    prob_map = probs_np[i]

                    # Apply Optimized Threshold
                    binary_mask = (prob_map > best_threshold).astype(np.uint8)
                    rle = rle_encode(binary_mask)

                    submission_ids.append(img_id)
                    submission_rles.append(rle)

        # Save
        sub_df = pd.DataFrame({"id": submission_ids, "rle_mask": submission_rles})
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation Metric {best_global_map:.5f} did not meet threshold 0.827. Skipping submission."
        )


if __name__ == "__main__":
    main()
