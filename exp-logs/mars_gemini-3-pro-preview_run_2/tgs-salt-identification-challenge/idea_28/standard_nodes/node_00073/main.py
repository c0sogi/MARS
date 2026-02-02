import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import (
    prepare_train_data,
    prepare_test_data,
    get_dataloaders,
    get_test_dataloader,
    SaltDataset,
)
from library.model import SaltNet
from library.losses import SegmentationLoss
from library.engine import (
    set_seed,
    train_one_epoch,
    evaluate,
    predict_tta,
    generate_submission,
)
from library.utils import optimize_threshold, calc_iou


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    set_seed(Config.SEED)
    device = Config.DEVICE

    print(f"Starting Run. Device: {device}, Epochs: {Config.EPOCHS}")

    # Ensure data is prepared and cached
    train_meta_df = prepare_train_data(load_cached_data=True)

    # Create directories for artifacts
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    model_dir = os.path.join(Config.WORKING_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Train 5-Fold Ensemble (Supervised Only)
    # -------------------------------------------------------------------------
    print("\n=== Training 5-Fold Ensemble ===")

    fold_models = []
    fold_metrics = []
    fold_thresholds = []

    # Validation data for final analysis (using Fold 0 as proxy or aggregate)
    # We will aggregate OOF predictions for final metric
    oof_preds = []
    oof_targets = []
    oof_ids = []

    for fold in range(Config.NUM_FOLDS):
        print(f"\nTraining Fold {fold}/{Config.NUM_FOLDS - 1}")

        # Get dataloaders for this fold
        train_loader, val_loader = get_dataloaders(fold=fold, load_cached_data=True)

        # Initialize Model, Loss, Optimizer
        model = SaltNet().to(device)
        criterion = SegmentationLoss()
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

        best_map = 0.0
        best_th = 0.5
        best_model_path = os.path.join(model_dir, f"model_fold_{fold}.pth")

        # Training Loop
        for epoch in range(1, Config.EPOCHS + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            # Save best model based on optimized mAP (Cite solution_lesson_node_00033)
            if val_metrics["map"] > best_map:
                best_map = val_metrics["map"]
                best_th = val_metrics["threshold"]
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} Best mAP: {best_map:.4f} at Threshold {best_th:.3f}")
        fold_metrics.append(best_map)
        fold_thresholds.append(best_th)
        fold_models.append(best_model_path)

        # Generate OOF predictions for this fold using best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # We need to collect OOF data for final analysis
        # Using predict_tta logic but for validation loader
        # Crop indices
        start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
        end = start + Config.ORIG_SIZE

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    images, masks, depths, ids = batch
                else:
                    continue
                images = images.to(device)
                depths = depths.to(device)
                masks = masks.to(device)

                # TTA Prediction
                logits = model(images, depths)
                probs = torch.sigmoid(logits)

                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, dims=[3])

                avg_probs = (probs + probs_flip) / 2.0

                # Crop
                avg_probs_crop = avg_probs[:, 0, start:end, start:end]
                masks_cpu = masks.cpu().numpy()[:, 0, start:end, start:end]

                oof_preds.append(avg_probs_crop.cpu().numpy())
                oof_targets.append(masks_cpu)
                oof_ids.extend(ids)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. Final Analysis on OOF Data
    # -------------------------------------------------------------------------
    print("\n=== Final Validation & Failure Analysis ===")

    all_preds = np.concatenate(oof_preds, axis=0)
    all_targets = np.concatenate(oof_targets, axis=0)

    # Optimize Global Threshold
    best_th, best_score = optimize_threshold(all_preds, all_targets)
    print(f"Global Optimal Threshold: {best_th}, Score: {best_score:.5f}")

    final_metric = best_score
    print(f"Final Validation Metric: {final_metric:.10f}")

    # Failure Analysis
    errors = []
    for i in range(len(all_preds)):
        p = (all_preds[i] > best_th).astype(np.uint8)
        t = all_targets[i].astype(np.uint8)
        iou = calc_iou(p, t)
        errors.append(1.0 - iou)

    id_to_error = dict(zip(oof_ids, errors))

    # Reconstruct dataframe for analysis
    analysis_df = train_meta_df[train_meta_df["id"].isin(oof_ids)].copy()
    analysis_df["error"] = analysis_df["id"].map(id_to_error)
    analysis_df = analysis_df.dropna(subset=["error"])

    corr_depth = analysis_df["z"].corr(analysis_df["error"])
    corr_salt = analysis_df["salt_coverage"].corr(analysis_df["error"])

    print("\nFailure Analysis:")
    print(f"Correlation (Error vs Depth): {corr_depth:.4f}")
    print(f"Correlation (Error vs Salt Coverage): {corr_salt:.4f}")

    # -------------------------------------------------------------------------
    # 4. Submission
    # -------------------------------------------------------------------------
    if final_metric > 0.7985:
        print("\nMetric condition met. Generating submission...")

        test_loader = get_test_dataloader(load_cached_data=True)
        avg_preds = None

        # Ensemble Inference
        for idx, model_path in enumerate(fold_models):
            print(f"Predicting with model {idx}...")
            model = SaltNet().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            preds, ids = predict_tta(model, test_loader, device)

            if avg_preds is None:
                avg_preds = preds
            else:
                avg_preds += preds

            del model
            torch.cuda.empty_cache()

        avg_preds /= len(fold_models)

        # Binarize with global threshold
        if avg_preds.ndim == 4:
            avg_preds = avg_preds[:, 0, :, :]

        binary_preds = (avg_preds > best_th).astype(np.uint8)

        # Encode
        from library.utils import rle_encode

        rle_masks = []
        for i in range(len(ids)):
            rle = rle_encode(binary_preds[i])
            rle_masks.append(rle)

        submission_df = pd.DataFrame({"id": ids, "rle_mask": rle_masks})
        output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")

    else:
        print(f"\nMetric {final_metric:.4f} <= 0.7985. Skipping submission.")


if __name__ == "__main__":
    main()
