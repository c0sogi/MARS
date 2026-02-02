import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import (
    seed_everything,
    rle_encode,
    calculate_map,
    center_crop,
    AverageMeter,
)
from library.model import SaltUNetPlusPlus
from library.data import get_loaders, get_test_loader
from library.engine import train_one_epoch, validate_one_epoch
from library.losses import get_loss

# =============================================================================
# Configuration Override for Fast Baseline
# =============================================================================
# Override Config values to ensure execution within 1 hour
Config.EPOCHS = 12
Config.PHASE1_EPOCHS = 6
Config.N_FOLDS = 5  # Keep 5 folds for robust ensemble
Config.BATCH_SIZE = 32  # Safe batch size for 128x128 on A100


def predict_val(model, loader, device):
    """
    Generates predictions for the validation set.
    Returns:
        probs (np.array): (N, H, W) probabilities
        targets (np.array): (N, H, W) ground truth
        depths (np.array): (N,) depths
    """
    model.eval()
    probs_list = []
    targets_list = []
    depths_list = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)

            # Forward pass (Phase 2 mode: single output)
            outputs = model(inputs, deep_supervision=False)

            # Sigmoid
            batch_probs = torch.sigmoid(outputs)

            # Crop to original size
            batch_probs = center_crop(
                batch_probs, Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG
            )
            batch_targets = center_crop(
                targets, Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG
            )

            probs_list.append(batch_probs.squeeze(1).cpu().numpy())
            targets_list.append(batch_targets.squeeze(1).cpu().numpy())

            # Extract depth from input tensor (channel 2)
            # Input is (B, 3, H, W). Channel 2 is depth channel.
            # We take the mean of the channel to get the scalar depth value (normalized)
            d = inputs[:, 2, :, :].mean(dim=(1, 2)).cpu().numpy()
            depths_list.append(d)

    return (
        np.concatenate(probs_list),
        np.concatenate(targets_list),
        np.concatenate(depths_list),
    )


def optimize_threshold(probs, targets):
    """
    Sweeps pixel probability thresholds to maximize mAP.
    """
    best_thr = 0.5
    best_score = -1

    # Sweep range
    thresholds = np.arange(0.3, 0.75, 0.05)

    print("Optimizing pixel threshold...")
    for thr in thresholds:
        # Binarize predictions based on current threshold
        preds_bin = (probs > thr).astype(np.float32)

        # Calculate mAP (competition metric)
        # calculate_map expects binary inputs or probabilities.
        # If we pass binary, it treats > 0.5 as 1.
        score = calculate_map(preds_bin, targets)

        if score > best_score:
            best_score = score
            best_thr = thr

    return best_thr, best_score


def failure_analysis(probs, targets, depths, best_thr):
    """
    Analyzes correlation between errors and metadata.
    """
    print("\n--- Failure Analysis ---")

    # Calculate per-image mAP
    scores = []
    coverages = []

    # Threshold predictions
    preds_bin = (probs > best_thr).astype(np.float32)

    thresholds_iou = np.arange(0.5, 0.96, 0.05)

    for i in range(len(preds_bin)):
        p = preds_bin[i]
        t = targets[i]

        # Calculate IoU
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        iou = intersection / (union + 1e-6) if (p.sum() + t.sum()) > 0 else 1.0

        # Calculate mAP for this image
        matches = iou > thresholds_iou
        scores.append(np.mean(matches))

        # Calculate coverage
        coverages.append(t.mean())

    scores = np.array(scores)
    coverages = np.array(coverages)
    depths = np.array(depths)

    # Correlation
    # We look for correlation with Error (1 - score)
    errors = 1.0 - scores

    corr_depth, _ = pearsonr(depths, errors)
    corr_cov, _ = pearsonr(coverages, errors)

    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_cov:.4f}")

    return scores


def main():
    seed_everything(Config.SEED)

    # Setup
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    oof_probs = []
    oof_targets = []
    oof_depths = []

    # =========================================================================
    # Training Loop (5 Folds)
    # =========================================================================
    for fold in range(Config.N_FOLDS):
        print(f"\n=== Training Fold {fold} ===")

        # Data Loaders
        train_loader, val_loader = get_loaders(fold, load_cached_data=True)

        # Model
        model = SaltUNetPlusPlus(deep_supervision=True).to(device)

        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=False,
        )

        # Scaler
        scaler = GradScaler()

        best_map = 0.0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}_best.pth")

        # Epoch Loop
        for epoch in range(1, Config.EPOCHS + 1):
            # Determine Phase
            if epoch <= Config.PHASE1_EPOCHS:
                phase = "phase1"
            else:
                phase = "phase2"

            loss_fn = get_loss(phase)

            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, loss_fn, device, epoch, phase
            )

            # Validate
            val_loss, val_map = validate_one_epoch(
                model, val_loader, loss_fn, device, phase
            )

            # Scheduler Step
            scheduler.step(val_map)

            # Save Best
            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), best_model_path)

        # Load Best Model for OOF
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Generate OOF Predictions
        probs, targets, depths = predict_val(model, val_loader, device)
        oof_probs.append(probs)
        oof_targets.append(targets)
        oof_depths.append(depths)

        # Cleanup
        del model, optimizer, scaler, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # Global Evaluation
    # =========================================================================
    print("\n=== Global Evaluation ===")
    oof_probs = np.concatenate(oof_probs)
    oof_targets = np.concatenate(oof_targets)
    oof_depths = np.concatenate(oof_depths)

    # Optimize Threshold
    best_thr, final_metric = optimize_threshold(oof_probs, oof_targets)

    print(f"Best Pixel Threshold: {best_thr:.4f}")
    print(f"Final Validation Metric: {final_metric:.15f}")

    # Failure Analysis
    failure_analysis(oof_probs, oof_targets, oof_depths, best_thr)

    # =========================================================================
    # Submission
    # =========================================================================
    if final_metric > 0.827:
        print("\nMetric > 0.827. Generating Submission...")

        test_loader, test_ids = get_test_loader(load_cached_data=True)

        # Initialize ensemble predictions
        # Shape: (N_test, H, W)
        ensemble_probs = np.zeros(
            (len(test_ids), Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG),
            dtype=np.float32,
        )

        for fold in range(Config.N_FOLDS):
            print(f"Inference Fold {fold}...")
            model = SaltUNetPlusPlus(deep_supervision=False).to(device)
            model.load_state_dict(
                torch.load(
                    os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}_best.pth"),
                    map_location=device,
                )
            )
            model.eval()

            fold_probs = []

            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(device)

                    # TTA: Original
                    out = model(inputs, deep_supervision=False)
                    prob = torch.sigmoid(out)

                    # TTA: Horizontal Flip
                    inputs_flip = torch.flip(inputs, [3])
                    out_flip = model(inputs_flip, deep_supervision=False)
                    prob_flip = torch.sigmoid(out_flip)
                    prob_flip = torch.flip(prob_flip, [3])

                    # Average TTA
                    avg_prob = (prob + prob_flip) / 2.0

                    # Crop
                    avg_prob = center_crop(
                        avg_prob, Config.IMG_HEIGHT_ORIG, Config.IMG_WIDTH_ORIG
                    )
                    fold_probs.append(avg_prob.squeeze(1).cpu().numpy())

            ensemble_probs += np.concatenate(fold_probs)

            del model
            torch.cuda.empty_cache()

        # Average Ensemble
        ensemble_probs /= Config.N_FOLDS

        # Apply Threshold
        pred_masks = (ensemble_probs > best_thr).astype(np.uint8)

        # Encode
        rle_list = []
        for mask in pred_masks:
            rle_list.append(rle_encode(mask))

        # Save
        sub_df = pd.DataFrame({"id": test_ids, "rle_mask": rle_list})
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric:.4f} <= 0.827. Skipping submission.")


if __name__ == "__main__":
    main()
